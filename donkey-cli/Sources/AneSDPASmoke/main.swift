// Donkey kernel #4 — SDPA (scaled dot-product attention).
//
// Input: [1, 3*DIM, 1, SP] = Q|K|V concatenated along channels.
// Output: [1, DIM, 1, SP] = pre-Wo attention output.
//
// Math: aw = softmax((Q·K^T) / sqrt(HD) + causal_mask),  y = aw·V
//
// Per-head shape pipeline matches maderix's gen_sdpa_fwd_taps:
//   [1, DIM, 1, SP] -> reshape [1, HEADS, HD, SP] -> transpose [1, HEADS, SP, HD]
//
// Donkey v1 shapes: DIM=1024, HEADS=16, HD=64, SP=K_PAD=8.
// Causal mask is fixed [SP, SP] lower-triangular, baked into weight blob.
// (For K<K_PAD active positions, masking positions >=K is handled by the
// inference loop, not this kernel — kernel always runs the full SP.)
import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(dim: Int, heads: Int, hd: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-trainer/kernels/sdpa.mil.template")
    let tmpl = try String(contentsOf: url, encoding: .utf8)
    let scale = 1.0 / Float(hd).squareRoot()
    return tmpl
        .replacingOccurrences(of: "{DIM3}",  with: String(3 * dim))
        .replacingOccurrences(of: "{DIM2}",  with: String(2 * dim))
        .replacingOccurrences(of: "{DIM}",   with: String(dim))
        .replacingOccurrences(of: "{HEADS}", with: String(heads))
        .replacingOccurrences(of: "{HD}",    with: String(hd))
        .replacingOccurrences(of: "{SP}",    with: String(sp))
        .replacingOccurrences(of: "{SCALE}", with: String(format: "%f", scale))
}

fileprivate func rng(_ seed: inout UInt32) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    let bits = (seed >> 8) & 0x00FFFFFF
    return Float(bits) / Float(1 << 23) - 1.0
}
fileprivate func randomArray(count: Int, seed: UInt32, scale: Float = 0.1) -> [Float] {
    var s = seed
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count { a[i] = rng(&s) * scale }
    return a
}

// Build the causal mask blob: [SP, SP] fp16, lower-tri = 0, upper-tri = -65504.
fileprivate func buildCausalMaskBlob(sp: Int) -> Data {
    var mask = [Float](repeating: 0, count: sp * sp)
    for r in 0..<sp {
        for c in 0..<sp {
            mask[r * sp + c] = (c <= r) ? 0.0 : -65504.0
        }
    }
    // Use aneBuildWeightBlobFP16 with rows=SP, cols=SP — same layout convention.
    return aneBuildWeightBlobFP16(mask, rows: sp, cols: sp)
}

// CPU reference SDPA. Operates on the same [1, 3*DIM, 1, SP] concat input
// to match the kernel exactly. Output [DIM, SP] row-major (channels-major
// per IOSurface convention).
fileprivate func sdpaRef(qkv: [Float], dim: Int, heads: Int, hd: Int, sp: Int) -> [Float] {
    let scale = 1.0 / Float(hd).squareRoot()

    // Slice Q, K, V; each is [DIM, SP] row-major.
    let qOff = 0,         kOff = dim * sp, vOff = 2 * dim * sp
    func get(_ buf: Int, _ ch: Int, _ s: Int) -> Float {
        return qkv[buf + ch * sp + s]
    }

    var out = [Float](repeating: 0, count: dim * sp)

    for h in 0..<heads {
        // Per-head scores [SP, SP]: scores[i, j] = (1/sqrt(HD)) * sum_d Q[h,i,d] * K[h,j,d]
        // Q[h, i, d] lives at channel (h*HD + d), spatial i.
        var scores = [Float](repeating: 0, count: sp * sp)
        for i in 0..<sp {
            for j in 0..<sp {
                var s: Float = 0
                for d in 0..<hd {
                    let ch = h * hd + d
                    s += get(qOff, ch, i) * get(kOff, ch, j)
                }
                scores[i * sp + j] = s * scale
                // Causal mask
                if j > i { scores[i * sp + j] = -65504.0 }
            }
        }

        // Softmax over last dim (j) per row i.
        for i in 0..<sp {
            // max for numerical stability
            var mx: Float = -Float.infinity
            for j in 0..<sp { if scores[i * sp + j] > mx { mx = scores[i * sp + j] } }
            var sum: Float = 0
            for j in 0..<sp {
                let e = exp(scores[i * sp + j] - mx)
                scores[i * sp + j] = e
                sum += e
            }
            for j in 0..<sp { scores[i * sp + j] /= sum }
        }

        // Apply attention to V: y[h, i, d] = sum_j scores[i, j] * V[h, j, d]
        for i in 0..<sp {
            for d in 0..<hd {
                var s: Float = 0
                for j in 0..<sp {
                    let ch = h * hd + d
                    s += scores[i * sp + j] * get(vOff, ch, j)
                }
                let outCh = h * hd + d
                out[outCh * sp + i] = s
            }
        }
    }
    return out
}

fileprivate func cosSim(_ a: [Float], _ b: [Float]) -> Float {
    var d: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    return d / (sqrt(na)*sqrt(nb) + 1e-12)
}

@main
struct AneSDPASmoke {
    static func main() {
        let DIM   = 1024
        let HEADS = 16
        let HD    = 64
        let SP    = Int(ProcessInfo.processInfo.environment["SP"] ?? "16") ?? 16     // K_PAD=16 (ANE SP minimum for SDPA graphs; SP=8 fails)
        slog("[sdpa] DIM=\(DIM) HEADS=\(HEADS) HD=\(HD) SP=\(SP)")

        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }

        // Mask blob.
        let maskBlob = buildCausalMaskBlob(sp: SP)
        slog("[sdpa] mask blob: \(maskBlob.count) bytes (expect \(128 + SP * SP * 2))")

        let mil: String
        do { mil = try loadMIL(dim: DIM, heads: HEADS, hd: HD, sp: SP) } catch {
            slog("template FAIL: \(error)"); exit(2)
        }

        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [("@model_path/weights/weight.bin", maskBlob)],
                inputBytes:  [3 * DIM * SP * 4],
                outputBytes: [DIM * SP * 4]
            )
        } catch { slog("compile FAIL: \(error)"); exit(3) }
        slog("[sdpa] compiled, compile_count=\(aneCompileCount())")

        let qkv = randomArray(count: 3 * DIM * SP, seed: 0xBADC_AFE, scale: 0.1)

        do {
            try qkv.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            try kernel.eval()
        } catch { slog("eval FAIL: \(error)"); exit(4) }

        var out = [Float](repeating: 0, count: DIM * SP)
        do {
            try out.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
        } catch { slog("read FAIL: \(error)"); exit(5) }

        let expected = sdpaRef(qkv: qkv, dim: DIM, heads: HEADS, hd: HD, sp: SP)
        let cos = cosSim(out, expected)

        var maxAbsErr: Float = 0
        for i in 0..<out.count {
            let e = abs(out[i] - expected[i])
            if e > maxAbsErr { maxAbsErr = e }
        }

        slog("[sdpa] out[0..4]:      \(Array(out[0..<4]))")
        slog("[sdpa] expected[0..4]: \(Array(expected[0..<4]))")
        slog("[sdpa] cosine sim:     \(cos)")
        slog("[sdpa] max abs err:    \(maxAbsErr)")

        if cos < 0.99 {
            slog("[sdpa] FAIL: cosine \(cos) < 0.99")
            exit(6)
        }

        // Steady-state latency: warmup + 100 iters.
        for _ in 0..<10 {
            do {
                try qkv.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
                try kernel.eval()
            } catch { slog("warmup FAIL"); exit(7) }
        }
        let t0 = Date()
        for _ in 0..<100 {
            do {
                try qkv.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
                try kernel.eval()
            } catch { slog("bench FAIL"); exit(8) }
        }
        let perCall = -t0.timeIntervalSinceNow * 1000.0 / 100.0
        slog("[sdpa] steady-state: \(String(format: "%.3f ms/call (write+eval, over 100 iters)", perCall))")
        slog("[sdpa] OK")
    }
}
