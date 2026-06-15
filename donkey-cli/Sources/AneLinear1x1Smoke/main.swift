// Donkey kernel #2 — parametric 1x1 conv (linear1x1).
//
// One MIL template, instantiated per (IN_CH, OUT_CH, SP) combination with its
// own weight blob. v1 uses 12 instances across the model:
//   Layer 0/1 Q, K, V, O:  1024 -> 1024     (8 instances)
//   Layer 0/1 FFN up:      1024 -> 4096     (2 instances)
//   Layer 0/1 FFN down:    4096 -> 1024     (2 instances)
//   Output head:           1024 -> 4097     (1 instance)
//   Token projection:      4096 -> 1024     (1 instance, separate from FFN down)
//   tap_fuse already handled separately.
//
// This smoke verifies the template works at all four distinct shapes by
// compiling and evaluating each in turn against a CPU reference.
import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(inCh: Int, outCh: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-trainer/kernels/linear1x1.mil.template")
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

fileprivate func cosSim(_ a: [Float], _ b: [Float]) -> Float {
    var d: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    return d / (sqrt(na)*sqrt(nb) + 1e-12)
}

// Compile + eval + validate one shape. Returns eval time in ms.
fileprivate func runShape(label: String, inCh: Int, outCh: Int, sp: Int,
                          wSeed: UInt32, xSeed: UInt32) -> Double {
    slog("--- \(label): IN=\(inCh) OUT=\(outCh) SP=\(sp) ---")

    let W = randomArray(count: outCh * inCh, seed: wSeed)
    let blob = aneBuildWeightBlobFP16(W, rows: outCh, cols: inCh)

    let mil: String
    do { mil = try loadMIL(inCh: inCh, outCh: outCh, sp: sp) } catch {
        slog("  template FAIL: \(error)"); exit(2)
    }

    let kernel: ANEKernel
    do {
        kernel = try aneCompile(
            milText: mil,
            weights: [("@model_path/weights/weight.bin", blob)],
            inputBytes:  [inCh * sp * 4],
            outputBytes: [outCh * sp * 4]
        )
    } catch { slog("  compile FAIL: \(error)"); exit(3) }
    slog("  compiled (count=\(aneCompileCount()))")

    let x = randomArray(count: inCh * sp, seed: xSeed)

    let t0 = Date()
    do {
        try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
        try kernel.eval()
    } catch { slog("  eval FAIL: \(error)"); exit(4) }
    let evalMs = -t0.timeIntervalSinceNow * 1000.0  // wall time including I/O

    var out = [Float](repeating: 0, count: outCh * sp)
    do {
        try out.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
    } catch { slog("  read FAIL: \(error)"); exit(5) }

    let expected = matmulRef(W: W, x: x, outCh: outCh, inCh: inCh, sp: sp)
    let cos = cosSim(out, expected)
    let gflops = 2.0 * Double(inCh) * Double(outCh) * Double(sp) / 1e9
    slog("  cosine: \(cos)  |  \(String(format: "%.2f GFLOPs in wall+IO %.2f ms", gflops, evalMs))")
    if cos < 0.999 {
        slog("  FAIL: cosine \(cos) < 0.999")
        slog("  out[0..4]:      \(Array(out[0..<4]))")
        slog("  expected[0..4]: \(Array(expected[0..<4]))")
        exit(6)
    }
    return evalMs
}

@main
struct AneLinear1x1Smoke {
    static func main() {
        slog("[linear1x1] parametric 1x1 conv smoke — 4 shapes")
        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }

        let SP = 64

        // Different seeds per shape so collisions don't paper over bugs.
        let qkvo = runShape(label: "Q/K/V/O proj (1024->1024)",
                            inCh: 1024, outCh: 1024, sp: SP,
                            wSeed: 0xA001, xSeed: 0xB001)
        let ffnUp = runShape(label: "FFN up (1024->4096)",
                             inCh: 1024, outCh: 4096, sp: SP,
                             wSeed: 0xA002, xSeed: 0xB002)
        let ffnDn = runShape(label: "FFN down (4096->1024)",
                             inCh: 4096, outCh: 1024, sp: SP,
                             wSeed: 0xA003, xSeed: 0xB003)
        let head = runShape(label: "Output head (1024->4097)",
                            inCh: 1024, outCh: 4097, sp: SP,
                            wSeed: 0xA004, xSeed: 0xB004)

        slog("")
        slog("[linear1x1] summary (wall+IO ms per call):")
        slog("  Q/K/V/O 1024->1024:  \(String(format: "%.2f ms", qkvo))")
        slog("  FFN up  1024->4096:  \(String(format: "%.2f ms", ffnUp))")
        slog("  FFN dn  4096->1024:  \(String(format: "%.2f ms", ffnDn))")
        slog("  Head    1024->4097:  \(String(format: "%.2f ms", head))")
        slog("[linear1x1] total compile count: \(aneCompileCount())")
        slog("[linear1x1] OK")
    }
}
