// OpsSmoke — verify DonkeyOps module: links cleanly, all 4 ops behave
// the same as the validated CpuXxxSmoke implementations.
import Foundation
import DonkeyOps

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func rng(_ seed: inout UInt32) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    return Float((seed >> 8) & 0x00FFFFFF) / Float(1 << 23) - 1.0
}
fileprivate func randomArray(count: Int, seed: UInt32, scale: Float = 1.0) -> [Float] {
    var s = seed
    var a = [Float](repeating: 0, count: count)
    for i in 0..<count { a[i] = rng(&s) * scale }
    return a
}

fileprivate func cosSim(_ a: [Float], _ b: [Float]) -> Float {
    var d: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    return d / (sqrt(na)*sqrt(nb) + 1e-12)
}

@main
struct OpsSmoke {
    static func main() {
        slog("[ops] verifying DonkeyOps module ...")

        // --- 1. RMSNorm ---
        let CH = 1024, SP = 16, EPS: Float = 1e-6
        var gamma = randomArray(count: CH, seed: 0xCAFE_FEED, scale: 0.1)
        for i in 0..<CH { gamma[i] += 1.0 }
        let x = randomArray(count: CH * SP, seed: 0xDEAD_BEEF, scale: 0.5)
        var yRms = [Float](repeating: 0, count: CH * SP)
        var scratchRms = [Float](repeating: 0, count: SP)
        x.withUnsafeBufferPointer { xb in
            gamma.withUnsafeBufferPointer { gb in
                yRms.withUnsafeMutableBufferPointer { yb in
                    cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                                out: yb.baseAddress!, ch: CH, sp: SP, eps: EPS,
                                scratch: &scratchRms)
                }
            }
        }
        // Naive reference.
        var yRef = [Float](repeating: 0, count: CH * SP)
        for s in 0..<SP {
            var ms: Float = 0
            for c in 0..<CH {
                let v = x[c * SP + s]
                ms += v * v
            }
            ms /= Float(CH)
            let rms = 1.0 / sqrt(ms + EPS)
            for c in 0..<CH {
                yRef[c * SP + s] = x[c * SP + s] * rms * gamma[c]
            }
        }
        let cosRms = cosSim(yRms, yRef)
        slog("  cpu_rmsnorm: cosine \(cosRms)")
        if cosRms < 0.99999 { slog("  FAIL"); exit(1) }

        // --- 2. SiLU ---
        let N = 4096 * 16
        let xs = randomArray(count: N, seed: 0xF00D_BEEF, scale: 2.0)
        var ySilu = [Float](repeating: 0, count: N)
        var scratchSilu = [Float](repeating: 0, count: 4096)
        xs.withUnsafeBufferPointer { xb in
            ySilu.withUnsafeMutableBufferPointer { yb in
                cpu_silu(x: xb.baseAddress!, out: yb.baseAddress!,
                         count: N, scratch: &scratchSilu)
            }
        }
        var ySiluRef = [Float](repeating: 0, count: N)
        for i in 0..<N { ySiluRef[i] = xs[i] / (1.0 + expf(-xs[i])) }
        let cosSilu = cosSim(ySilu, ySiluRef)
        slog("  cpu_silu: cosine \(cosSilu)")
        if cosSilu < 0.9999 { slog("  FAIL"); exit(2) }

        // --- 3. Residual add ---
        let a = randomArray(count: N, seed: 0xAAAA, scale: 0.5)
        let b = randomArray(count: N, seed: 0xBBBB, scale: 0.5)
        var ySum = [Float](repeating: 0, count: N)
        a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                ySum.withUnsafeMutableBufferPointer { yb in
                    cpu_residual_add(a: ab.baseAddress!, b: bb.baseAddress!,
                                     out: yb.baseAddress!, count: N)
                }
            }
        }
        var maxErr: Float = 0
        for i in 0..<N {
            let e = abs(ySum[i] - (a[i] + b[i]))
            if e > maxErr { maxErr = e }
        }
        slog("  cpu_residual_add: max err \(maxErr) (expect 0)")
        if maxErr > 1e-7 { slog("  FAIL"); exit(3) }

        // --- 4. Sigmoid scalar ---
        let sigOk = abs(cpu_sigmoid(0.0) - 0.5) < 1e-6
                 && abs(cpu_sigmoid(10.0) - 0.99995) < 1e-3
                 && abs(cpu_sigmoid(-10.0) - 0.0) < 1e-3
        slog("  cpu_sigmoid: \(sigOk ? "OK" : "FAIL")")
        if !sigOk { exit(4) }

        slog("[ops] all ops OK")
    }
}
