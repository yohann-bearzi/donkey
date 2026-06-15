// Microbenchmark: cost of round-tripping data between ANE and CPU.
//
// Anchor kernel: 1024x1024 conv (Q/K/V/O shape, ~0.45 ms eval).
// Three configurations measured:
//   A: ANE only — eval in a loop, no host touch between
//   B: ANE + read/write — eval, read output, write same data, eval again
//   C: ANE + vDSP RMSNorm — eval, read, rmsnorm, write, eval
//
// Per-call cost deltas tell us:
//   (B - A) = IOSurface read+write overhead per round-trip
//   (C - B) = pure CPU RMSNorm cost (we already measured 0.534 ms standalone)
//   (C - A) = total per-RMSNorm-site cost in a real donkey forward pass
//
// 5 RMSNorm sites in v1; this number tells us whether to keep RMSNorm on CPU
// or invest in fixing the ANE MIL form (reduce_sum pattern per maderix).
import Foundation
import Accelerate
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(ch: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-cli/Sources/AneRoundTripBench/rt.mil.template")
    let tmpl = try String(contentsOf: url, encoding: .utf8)
    return tmpl
        .replacingOccurrences(of: "{CH}", with: String(ch))
        .replacingOccurrences(of: "{SP}", with: String(sp))
}

fileprivate func rng(_ seed: inout UInt32) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    let bits = (seed >> 8) & 0x00FFFFFF
    return Float(bits) / Float(1 << 23) - 1.0
}
fileprivate func randomArray(count: Int, seed: UInt32, scale: Float = 1.0) -> [Float] {
    var s = seed
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count { a[i] = rng(&s) * scale }
    return a
}

// Same vDSP RMSNorm as the smoke target, inlined here.
fileprivate func rmsNormVDSP(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>,
                              gamma: [Float], ch: Int, sp: Int, eps: Float) {
    let invCh = Float(1.0) / Float(ch)
    var sq = [Float](repeating: 0, count: ch * sp)
    vDSP_vsq(input, 1, &sq, 1, vDSP_Length(ch * sp))
    var ss = [Float](repeating: 0, count: sp)
    for s in 0..<sp {
        var sum: Float = 0
        sq.withUnsafeBufferPointer { buf in
            vDSP_sve(buf.baseAddress! + s, sp, &sum, vDSP_Length(ch))
        }
        ss[s] = sum * invCh + eps
    }
    var rrms = [Float](repeating: 0, count: sp)
    var n: Int32 = Int32(sp)
    vvrsqrtf(&rrms, ss, &n)
    for c in 0..<ch {
        let g = gamma[c]
        rrms.withUnsafeBufferPointer { rb in
            vDSP_vmul(input + c*sp, 1, rb.baseAddress!, 1, output + c*sp, 1, vDSP_Length(sp))
        }
        var gMut = g
        vDSP_vsmul(output + c*sp, 1, &gMut, output + c*sp, 1, vDSP_Length(sp))
    }
}

@main
struct AneRoundTripBench {
    static func main() {
        let CH = 1024
        let SP = 64
        let WARMUP = 10
        let ITERS  = 200
        let EPS: Float = 1e-6

        slog("=== ANE<->CPU round-trip microbenchmark ===")
        slog("Anchor kernel: \(CH)x\(CH) conv, SP=\(SP); \(ITERS) iters after \(WARMUP) warmup")

        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }

        let W = randomArray(count: CH * CH, seed: 0xA001)
        let blob = aneBuildWeightBlobFP16(W, rows: CH, cols: CH)
        let mil: String
        do { mil = try loadMIL(ch: CH, sp: SP) } catch { slog("template FAIL"); exit(2) }

        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [("@model_path/weights/weight.bin", blob)],
                inputBytes:  [CH * SP * 4],
                outputBytes: [CH * SP * 4]
            )
        } catch { slog("compile FAIL: \(error)"); exit(3) }

        // Initial input write.
        let x = randomArray(count: CH * SP, seed: 0xB001)
        do {
            try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
        } catch { slog("initial write FAIL: \(error)"); exit(4) }

        // ----------------------------------------------------------------
        // Configuration A: ANE only, no host touch
        // ----------------------------------------------------------------
        for _ in 0..<WARMUP {
            do { try kernel.eval() } catch { slog("A warmup FAIL"); exit(5) }
        }
        let tA = Date()
        for _ in 0..<ITERS {
            do { try kernel.eval() } catch { slog("A FAIL"); exit(6) }
        }
        let dA = -tA.timeIntervalSinceNow * 1000.0
        let perA = dA / Double(ITERS)
        slog("A) eval only:                    \(String(format: "%.3f ms/call", perA))")

        // ----------------------------------------------------------------
        // Configuration B: eval -> read -> write -> eval (no compute)
        // ----------------------------------------------------------------
        var buf = [Float](repeating: 0, count: CH * SP)
        for _ in 0..<WARMUP {
            do {
                try kernel.eval()
                try buf.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
                try buf.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            } catch { slog("B warmup FAIL: \(error)"); exit(7) }
        }
        let tB = Date()
        for _ in 0..<ITERS {
            do {
                try kernel.eval()
                try buf.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
                try buf.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            } catch { slog("B FAIL: \(error)"); exit(8) }
        }
        let dB = -tB.timeIntervalSinceNow * 1000.0
        let perB = dB / Double(ITERS)
        slog("B) eval + read + write:          \(String(format: "%.3f ms/call", perB))")

        // ----------------------------------------------------------------
        // Configuration C: eval -> read -> vDSP RMSNorm -> write -> eval
        // ----------------------------------------------------------------
        var gamma = randomArray(count: CH, seed: 0xC001, scale: 0.1)
        for i in 0..<CH { gamma[i] += 1.0 }
        var rmsOut = [Float](repeating: 0, count: CH * SP)

        for _ in 0..<WARMUP {
            do {
                try kernel.eval()
                try buf.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
                buf.withUnsafeBufferPointer { bb in
                    rmsOut.withUnsafeMutableBufferPointer { rb in
                        rmsNormVDSP(input: bb.baseAddress!, output: rb.baseAddress!,
                                    gamma: gamma, ch: CH, sp: SP, eps: EPS)
                    }
                }
                try rmsOut.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            } catch { slog("C warmup FAIL: \(error)"); exit(9) }
        }
        let tC = Date()
        for _ in 0..<ITERS {
            do {
                try kernel.eval()
                try buf.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
                buf.withUnsafeBufferPointer { bb in
                    rmsOut.withUnsafeMutableBufferPointer { rb in
                        rmsNormVDSP(input: bb.baseAddress!, output: rb.baseAddress!,
                                    gamma: gamma, ch: CH, sp: SP, eps: EPS)
                    }
                }
                try rmsOut.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            } catch { slog("C FAIL: \(error)"); exit(10) }
        }
        let dC = -tC.timeIntervalSinceNow * 1000.0
        let perC = dC / Double(ITERS)
        slog("C) eval + read + rmsnorm + write: \(String(format: "%.3f ms/call", perC))")

        slog("")
        slog("=== Cost decomposition (per round-trip) ===")
        slog("  Pure ANE eval (A):                  \(String(format: "%.3f ms", perA))")
        slog("  IOSurface read+write (B - A):       \(String(format: "%.3f ms", perB - perA))")
        slog("  vDSP RMSNorm proper (C - B):        \(String(format: "%.3f ms", perC - perB))")
        slog("  TOTAL per RMSNorm site (C - A):     \(String(format: "%.3f ms", perC - perA))")
        slog("")
        slog("=== Implication for donkey v1 (5 RMSNorm sites) ===")
        slog("  CPU RMSNorm overhead per fwd pass:  \(String(format: "%.2f ms", 5.0 * (perC - perA)))")
        slog("  (vs. estimated ANE RMSNorm via maderix reduce_sum pattern: ~5 x 0.3 ms = 1.5 ms)")
    }
}
