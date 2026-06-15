// Donkey kernel #1 — input adapter (tap fusion).
//
// Mathematically: g = concat([tap_lo, tap_mid, tap_hi]) @ W_full
// where W_full is the concatenation of [W_lo; W_mid; W_hi] along input channels.
//
// Implementation: CPU concatenates the three taps into one [1, 3*IN_CH, 1, SP]
// IOSurface; ANE runs a single conv (3*IN_CH -> OUT_CH) against one concatenated
// weight. This is the canonical EAGLE-3 formulation, and works around a bug we
// found in the multi-input path of ane_bridge_compile_multi_weights (when 3
// separate IOSurfaces are bound, only one of them gets data through reliably;
// the others read garbage that's deterministic but unrelated to what we wrote).
//
// TODO(bridge): investigate why ane_bridge_compile_multi_weights produces
// incorrect results with 3 input IOSurfaces. The maderix repo never exercises
// multi-input MIL on the bridge, so we're first to hit it. For v1 we route
// around; the v2 training path may need it for backward pass dx outputs.
import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(inChTotal: Int, outCh: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-trainer/kernels/tap_fuse.mil.template")
    let tmpl = try String(contentsOf: url, encoding: .utf8)
    return tmpl
        .replacingOccurrences(of: "{IN_CH_TOTAL}", with: String(inChTotal))
        .replacingOccurrences(of: "{OUT_CH}", with: String(outCh))
        .replacingOccurrences(of: "{SP}", with: String(sp))
}

fileprivate func rng(_ seed: inout UInt32) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    let bits = (seed >> 8) & 0x00FFFFFF
    return Float(bits) / Float(1 << 23) - 1.0
}
fileprivate func randomArray(count: Int, seed: UInt32, scale: Float = 0.05) -> [Float] {
    var s = seed
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count { a[i] = rng(&s) * scale }
    return a
}

// Concatenate three taps along the channel dim.
// Inputs:  three [IN_CH, SP] row-major arrays (channels-major per IOSurface).
// Output:  one [3*IN_CH, SP] row-major array, taps stacked in order [lo; mid; hi].
fileprivate func concatTapsByChannel(tap_lo: [Float], tap_mid: [Float], tap_hi: [Float],
                                     inCh: Int, sp: Int) -> [Float] {
    var out = [Float](repeating: 0, count: 3 * inCh * sp)
    // Copy each tap's [inCh, sp] row block into the output at the right offset.
    // tap_lo lives at channels [0, inCh), tap_mid at [inCh, 2*inCh), tap_hi at [2*inCh, 3*inCh).
    let stride = inCh * sp
    for i in 0..<stride { out[i]                = tap_lo[i]  }
    for i in 0..<stride { out[stride + i]       = tap_mid[i] }
    for i in 0..<stride { out[2 * stride + i]   = tap_hi[i]  }
    return out
}

// Concatenate three weight matrices along the input-channel dim.
// Inputs:  three [OUT_CH, IN_CH] row-major weight matrices.
// Output:  one [OUT_CH, 3*IN_CH] row-major weight matrix, weights interleaved
//          per output row: row c is [W_lo[c, :], W_mid[c, :], W_hi[c, :]].
fileprivate func concatWeightsByInputChannel(W_lo: [Float], W_mid: [Float], W_hi: [Float],
                                             outCh: Int, inCh: Int) -> [Float] {
    var W = [Float](repeating: 0, count: outCh * 3 * inCh)
    for c in 0..<outCh {
        let dstRow = c * 3 * inCh
        let srcRow = c * inCh
        for j in 0..<inCh { W[dstRow + j]              = W_lo[srcRow + j]  }
        for j in 0..<inCh { W[dstRow + inCh + j]       = W_mid[srcRow + j] }
        for j in 0..<inCh { W[dstRow + 2 * inCh + j]   = W_hi[srcRow + j]  }
    }
    return W
}

fileprivate func matmulRef(W: [Float], x: [Float], outCh: Int, inCh: Int, sp: Int) -> [Float] {
    var y = [Float](repeating: 0, count: outCh * sp)
    for c in 0..<outCh {
        for s in 0..<sp {
            var acc: Float = 0
            for cp in 0..<inCh { acc += W[c * inCh + cp] * x[cp * sp + s] }
            y[c * sp + s] = acc
        }
    }
    return y
}

@main
struct AneTapFuseSmoke {
    static func main() {
        let IN_CH       = 4096         // per-tap channel count (trunk hidden)
        let IN_CH_TOTAL = 3 * IN_CH    // = 12288 after channel-concat
        let OUT_CH      = 1024         // donkey hidden
        let SP          = 64           // safe SP (above ANE minimum)
        slog("[tap_fuse] IN_CH=\(IN_CH)x3=\(IN_CH_TOTAL) OUT_CH=\(OUT_CH) SP=\(SP)")

        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }
        slog("[tap_fuse] bridge ok")

        // Per-tap weights in maderix layout [out, in].
        let W_lo  = randomArray(count: OUT_CH * IN_CH, seed: 0x1111_1111)
        let W_mid = randomArray(count: OUT_CH * IN_CH, seed: 0x2222_2222)
        let W_hi  = randomArray(count: OUT_CH * IN_CH, seed: 0x3333_3333)

        // Concatenate into one [OUT_CH, 3*IN_CH] weight matrix.
        let W_full = concatWeightsByInputChannel(W_lo: W_lo, W_mid: W_mid, W_hi: W_hi,
                                                 outCh: OUT_CH, inCh: IN_CH)
        let blob = aneBuildWeightBlobFP16(W_full, rows: OUT_CH, cols: IN_CH_TOTAL)
        let expectedBlobBytes = 128 + OUT_CH * IN_CH_TOTAL * 2
        slog("[tap_fuse] W_full blob: \(blob.count) bytes (expect \(expectedBlobBytes))")

        let mil: String
        do { mil = try loadMIL(inChTotal: IN_CH_TOTAL, outCh: OUT_CH, sp: SP) } catch {
            slog("template FAIL: \(error)"); exit(2)
        }

        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [("@model_path/weights/weight.bin", blob)],
                inputBytes:  [IN_CH_TOTAL * SP * 4],
                outputBytes: [OUT_CH * SP * 4]
            )
        } catch { slog("compile FAIL: \(error)"); exit(3) }
        slog("[tap_fuse] compiled, compile_count=\(aneCompileCount())")

        // Three random taps.
        let tap_lo  = randomArray(count: IN_CH * SP, seed: 0xAAAA_AAAA)
        let tap_mid = randomArray(count: IN_CH * SP, seed: 0xBBBB_BBBB)
        let tap_hi  = randomArray(count: IN_CH * SP, seed: 0xCCCC_CCCC)

        // CPU-concat along channel dim into one input.
        let taps = concatTapsByChannel(tap_lo: tap_lo, tap_mid: tap_mid, tap_hi: tap_hi,
                                       inCh: IN_CH, sp: SP)

        do {
            try taps.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            try kernel.eval()
        } catch { slog("eval FAIL: \(error)"); exit(4) }

        var output = [Float](repeating: 0, count: OUT_CH * SP)
        do {
            try output.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
        } catch { slog("read FAIL: \(error)"); exit(5) }

        // Reference: same single concat-matmul on CPU.
        let expected = matmulRef(W: W_full, x: taps, outCh: OUT_CH, inCh: IN_CH_TOTAL, sp: SP)

        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<output.count {
            dot += output[i] * expected[i]; na += output[i]*output[i]; nb += expected[i]*expected[i]
        }
        let cos = dot / (sqrt(na)*sqrt(nb) + 1e-12)
        slog("[tap_fuse] out[0..4]:      \(Array(output[0..<4]))")
        slog("[tap_fuse] expected[0..4]: \(Array(expected[0..<4]))")
        slog("[tap_fuse] cosine sim:     \(cos)")

        if cos < 0.999 {
            slog("[tap_fuse] FAIL: cosine \(cos) < 0.999")
            exit(6)
        }
        slog("[tap_fuse] OK")
    }
}
