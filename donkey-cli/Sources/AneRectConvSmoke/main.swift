import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(inCh: Int, outCh: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-cli/Sources/AneRectConvSmoke/rect_conv.mil.template")
    let tmpl = try String(contentsOf: url, encoding: .utf8)
    return tmpl
        .replacingOccurrences(of: "{IN_CH}", with: String(inCh))
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

// W stored row-major [OUT_CH, IN_CH] per ane_mil_gen.h convention.
// x stored row-major [IN_CH, SP] (channels-major IOSurface layout).
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
struct AneRectConvSmoke {
    static func main() {
        let IN_CH = 4096, OUT_CH = 1024, SP = 64
        slog("[rect] IN_CH=\(IN_CH) OUT_CH=\(OUT_CH) SP=\(SP) -- single-weight non-square conv")

        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }

        let W = randomArray(count: OUT_CH * IN_CH, seed: 0xDEAD_BEEF)
        let blob = aneBuildWeightBlobFP16(W, rows: OUT_CH, cols: IN_CH)
        slog("[rect] blob: \(blob.count) bytes (expect \(128 + OUT_CH * IN_CH * 2))")

        let mil: String
        do { mil = try loadMIL(inCh: IN_CH, outCh: OUT_CH, sp: SP) } catch {
            slog("template FAIL: \(error)"); exit(2)
        }

        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [("@model_path/weights/weight.bin", blob)],
                inputBytes:  [IN_CH * SP * 4],
                outputBytes: [OUT_CH * SP * 4]
            )
        } catch { slog("compile FAIL: \(error)"); exit(3) }
        slog("[rect] compiled")

        let x = randomArray(count: IN_CH * SP, seed: 0xCAFE_BABE)
        do {
            try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            try kernel.eval()
        } catch { slog("eval FAIL: \(error)"); exit(4) }

        var out = [Float](repeating: 0, count: OUT_CH * SP)
        do {
            try out.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
        } catch { slog("read FAIL: \(error)"); exit(5) }

        let expected = matmulRef(W: W, x: x, outCh: OUT_CH, inCh: IN_CH, sp: SP)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<out.count {
            dot += out[i] * expected[i]; na += out[i] * out[i]; nb += expected[i] * expected[i]
        }
        let cos = dot / (sqrt(na) * sqrt(nb) + 1e-12)
        slog("[rect] out[0..4]:      \(Array(out[0..<4]))")
        slog("[rect] expected[0..4]: \(Array(expected[0..<4]))")
        slog("[rect] cosine sim:     \(cos)")
        if cos < 0.999 { slog("[rect] FAIL"); exit(6) }
        slog("[rect] OK")
    }
}
