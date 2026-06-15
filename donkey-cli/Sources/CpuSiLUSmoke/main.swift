// CPU SiLU via Accelerate vvexpf + vDSP_vmul.
// Measured replacement for ANE SiLU (which costs 0.508 ms/call at
// HIDDEN=4096 SP=16 — IO-dominated). CPU should land near 0.1 ms.
//
// Math: y = x * sigmoid(x) = x / (1 + exp(-x))
// vvexpf computes exp on a vector in one Accelerate call.
import Foundation
import Accelerate

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
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

// y = x * sigmoid(x). Four-pass naive version. KEPT for comparison only.
fileprivate func cpuSiLU_naive(_ x: [Float], _ out: inout [Float]) {
    let n = vDSP_Length(x.count)
    var negOne: Float = -1.0
    var one: Float = 1.0
    var neg = [Float](repeating: 0, count: x.count)
    vDSP_vsmul(x, 1, &negOne, &neg, 1, n)
    var nInt: Int32 = Int32(x.count)
    var e = [Float](repeating: 0, count: x.count)
    vvexpf(&e, neg, &nInt)
    vDSP_vsadd(e, 1, &one, &e, 1, n)
    vvrecf(&out, e, &nInt)
    vDSP_vmul(out, 1, x, 1, &out, 1, n)
}

// Fused SiLU — single pass over the data with one chunk-sized scratch.
//
// Strategy: process the input in cache-friendly chunks (CHUNK floats).
//   1. negate the chunk into scratch        (1 read,  1 write)
//   2. vvexpf in place on scratch           (1 r/w of scratch — in L1)
//   3. tight inner loop:                    (1 read x,  1 r/w scratch in L1,  1 write y)
//        y[i] = x[i] / (1 + scratch[i])
// Total memory traffic: ~2 reads + 1 write of full input = 3N (vs 8N naive).
// Scratch never leaves L1, so it costs no DRAM bandwidth.
fileprivate func cpuSiLU_fused(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                                count: Int) {
    let CHUNK = 4096   // 16KB, fits comfortably in L1D (192KB on M3)
    var scratch = [Float](repeating: 0, count: CHUNK)
    var pos = 0
    while pos < count {
        let n = min(CHUNK, count - pos)
        var nInt: Int32 = Int32(n)
        var negOne: Float = -1.0

        // scratch = -x  (one read of x, one write of scratch — scratch in L1)
        scratch.withUnsafeMutableBufferPointer { sb in
            vDSP_vsmul(x + pos, 1, &negOne, sb.baseAddress!, 1, vDSP_Length(n))
            // scratch = exp(scratch) in place
            vvexpf(sb.baseAddress!, sb.baseAddress!, &nInt)
        }
        // Inner loop fused:  y[i] = x[i] / (1 + scratch[i])
        // No vDSP combinator for this exact form; the loop autovectorizes
        // cleanly with -O on Swift release builds.
        for i in 0..<n {
            out[pos + i] = x[pos + i] / (1.0 + scratch[i])
        }
        pos += n
    }
}

// Wrapper that matches the array-in/array-out signature for the bench harness.
fileprivate func cpuSiLU(_ x: [Float], _ out: inout [Float]) {
    x.withUnsafeBufferPointer { xb in
        out.withUnsafeMutableBufferPointer { ob in
            cpuSiLU_fused(xb.baseAddress!, ob.baseAddress!, count: x.count)
        }
    }
}

fileprivate func siluNaive(_ x: [Float]) -> [Float] {
    var y = [Float](repeating: 0, count: x.count)
    for i in 0..<x.count {
        let s = 1.0 / (1.0 + exp(-x[i]))
        y[i] = x[i] * s
    }
    return y
}

fileprivate func cosSim(_ a: [Float], _ b: [Float]) -> Float {
    var d: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    return d / (sqrt(na)*sqrt(nb) + 1e-12)
}

@main
struct CpuSiLUSmoke {
    static func main() {
        let CH = 4096
        let SP = 16
        slog("[cpu_silu] CH=\(CH) SP=\(SP) (matches ANE SiLU shape)")

        let x = randomArray(count: CH * SP, seed: 0xF00D_BEEF, scale: 2.0)
        var y = [Float](repeating: 0, count: CH * SP)
        cpuSiLU(x, &y)

        let yRef = siluNaive(x)
        let cos = cosSim(y, yRef)
        var maxAbsErr: Float = 0
        for i in 0..<y.count {
            let e = abs(y[i] - yRef[i])
            if e > maxAbsErr { maxAbsErr = e }
        }
        slog("[cpu_silu] y[0..4]:      \(Array(y[0..<4]))")
        slog("[cpu_silu] yRef[0..4]:   \(Array(yRef[0..<4]))")
        slog("[cpu_silu] cosine sim:   \(cos)")
        slog("[cpu_silu] max abs err:  \(maxAbsErr)")

        if cos < 0.99999 {
            slog("[cpu_silu] FAIL: cosine \(cos) < 0.99999")
            exit(6)
        }

        // Bench naive (4-pass) for comparison.
        for _ in 0..<10 { cpuSiLU_naive(x, &y) }
        let t0n = Date()
        for _ in 0..<100 { cpuSiLU_naive(x, &y) }
        let perNaive = -t0n.timeIntervalSinceNow * 1000.0 / 100.0
        slog("[cpu_silu] naive (4-pass): \(String(format: "%.3f ms/call", perNaive))")

        // Bench fused.
        for _ in 0..<10 { cpuSiLU(x, &y) }
        let t0 = Date()
        for _ in 0..<100 { cpuSiLU(x, &y) }
        let perCall = -t0.timeIntervalSinceNow * 1000.0 / 100.0
        slog("[cpu_silu] fused:          \(String(format: "%.3f ms/call", perCall))")
        slog("[cpu_silu] speedup vs naive: \(String(format: "%.2fx", perNaive / perCall))")
        slog("[cpu_silu] vs ANE SiLU 0.508 ms -> \(String(format: "%.3f ms saved per site", 0.508 - perCall))")
        slog("[cpu_silu] OK")
    }
}
