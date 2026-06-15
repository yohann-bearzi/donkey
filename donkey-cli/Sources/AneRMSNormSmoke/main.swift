// Donkey kernel #3 — RMSNorm.
//
// Decision: RMSNorm runs on CPU via Accelerate vDSP. Matches maderix's
// proven path (stories_cpu_ops.h: cpu_rmsnorm + vvrsqrtf, 0.7ms for
// 768x512). His ANE rmsnorm (ane_classifier.h:gen_final_rmsnorm) exists
// only as PR#19 optional optimization to fuse with the classifier conv;
// we don't have that chain pressure. CPU-side saves 5 compile-budget
// slots (3 layer-norms in donkey's 2-layer model + 1 final, plus we'd
// need backward variants in v2).
//
// Implementation mirrors stories_cpu_ops.h::rmsnorm:
//   ss[s]  = sum_c (x[c,s] * x[c,s]) / D       (mean of squares)
//   rrms[s] = 1 / sqrt(ss[s] + eps)
//   out[c,s] = x[c,s] * rrms[s] * gamma[c]
//
// vDSP gives us: vDSP_vsq (square), vDSP_meanv (mean), vvrsqrtf (rsqrt),
// vDSP_vmul (per-channel scale). Per-spatial-position outer loop in Swift.
//
// This smoke target intentionally has NO ANE compile — it's pure-CPU
// validation that the math matches what we'll do at runtime. Kept under
// AneRMSNormSmoke directory for consistency, despite not touching ANE.
import Foundation
import Accelerate
import DonkeyANE  // for slog convention; ANE bridge not actually used

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

// Naive reference for validation (no vDSP, no tricks).
fileprivate func rmsNormNaive(x: [Float], gamma: [Float], ch: Int, sp: Int, eps: Float) -> [Float] {
    var y = [Float](repeating: 0, count: ch * sp)
    for s in 0..<sp {
        var ms: Float = 0
        for c in 0..<ch {
            let v = x[c * sp + s]
            ms += v * v
        }
        ms /= Float(ch)
        let rms = 1.0 / sqrt(ms + eps)
        for c in 0..<ch {
            y[c * sp + s] = x[c * sp + s] * rms * gamma[c]
        }
    }
    return y
}

// Production version: vDSP-vectorized, matches maderix's stories_cpu_ops.h
// layout exactly. Input is [CH, SP] row-major channels-major (matches our
// IOSurface convention). Operates fp32 throughout — accuracy not a concern
// at this scale, and we'd cast to fp16 at the next ANE input anyway.
fileprivate func rmsNormVDSP(x: [Float], gamma: [Float], ch: Int, sp: Int, eps: Float) -> [Float] {
    let invCh = Float(1.0) / Float(ch)
    var out = [Float](repeating: 0, count: ch * sp)
    let chU = vDSP_Length(ch)

    // Per-position: gather slice, square, mean, rsqrt, scale by gamma, write back.
    // Maderix's vectorized form processes all positions in parallel by squaring
    // the whole tensor first, then doing per-position mean. We do the same.
    var sq = [Float](repeating: 0, count: ch * sp)
    vDSP_vsq(x, 1, &sq, 1, vDSP_Length(ch * sp))  // sq = x*x

    // ss[s] = sum_c sq[c*sp + s] / ch
    // Layout has channel as the stride-sp dimension. For each spatial position s,
    // sum_c starts at offset s and strides by sp.
    var ss = [Float](repeating: 0, count: sp)
    for s in 0..<sp {
        var sum: Float = 0
        sq.withUnsafeBufferPointer { buf in
            vDSP_sve(buf.baseAddress! + s, sp, &sum, chU)
        }
        ss[s] = sum * invCh + eps
    }

    // rrms = 1 / sqrt(ss). vvrsqrtf is the vectorized rsqrt.
    var rrms = [Float](repeating: 0, count: sp)
    var n: Int32 = Int32(sp)
    vvrsqrtf(&rrms, ss, &n)

    // For each channel c, multiply x row-slice by rrms (per-position scale) and
    // by gamma[c] (scalar). Output row at offset c*sp.
    for c in 0..<ch {
        let g = gamma[c]
        x.withUnsafeBufferPointer { xb in
            rrms.withUnsafeBufferPointer { rb in
                out.withUnsafeMutableBufferPointer { ob in
                    // out[c*sp + s] = x[c*sp + s] * rrms[s] * g
                    // vDSP_vmul: out = x .* rrms
                    vDSP_vmul(xb.baseAddress! + c*sp, 1,
                              rb.baseAddress!,        1,
                              ob.baseAddress! + c*sp, 1,
                              vDSP_Length(sp))
                    // Then scale by g.
                    var gMut = g
                    vDSP_vsmul(ob.baseAddress! + c*sp, 1, &gMut,
                               ob.baseAddress! + c*sp, 1,
                               vDSP_Length(sp))
                }
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
struct AneRMSNormSmoke {
    static func main() {
        let CH = 1024
        let SP = 64
        let EPS: Float = 1e-6
        slog("[rmsnorm] CH=\(CH) SP=\(SP) eps=\(EPS) -- CPU via Accelerate vDSP (matches maderix)")

        var gamma = randomArray(count: CH, seed: 0xCAFE_FEED, scale: 0.1)
        for i in 0..<CH { gamma[i] += 1.0 }
        let x = randomArray(count: CH * SP, seed: 0xDEAD_BEEF, scale: 0.5)

        // Time the vDSP path.
        let warmupY = rmsNormVDSP(x: x, gamma: gamma, ch: CH, sp: SP, eps: EPS)
        _ = warmupY  // touch to avoid DCE

        let iters = 100
        let t0 = Date()
        var y = [Float]()
        for _ in 0..<iters {
            y = rmsNormVDSP(x: x, gamma: gamma, ch: CH, sp: SP, eps: EPS)
        }
        let elapsed = -t0.timeIntervalSinceNow * 1000.0
        let perCall = elapsed / Double(iters)

        let yRef = rmsNormNaive(x: x, gamma: gamma, ch: CH, sp: SP, eps: EPS)
        let cos = cosSim(y, yRef)

        var maxAbsErr: Float = 0
        for i in 0..<y.count {
            let e = abs(y[i] - yRef[i])
            if e > maxAbsErr { maxAbsErr = e }
        }

        slog("[rmsnorm] y[0..4]:      \(Array(y[0..<4]))")
        slog("[rmsnorm] yRef[0..4]:   \(Array(yRef[0..<4]))")
        slog("[rmsnorm] cosine sim:   \(cos)")
        slog("[rmsnorm] max abs err:  \(maxAbsErr)")
        slog("[rmsnorm] vDSP latency: \(String(format: "%.3f ms/call (over %d iters)", perCall, iters))")

        if cos < 0.99999 {
            slog("[rmsnorm] FAIL: cosine \(cos) < 0.99999")
            exit(6)
        }
        slog("[rmsnorm] OK")
    }
}
