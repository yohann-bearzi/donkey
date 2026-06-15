// Optimized CPU RMSNorm: single streaming pass over the input.
//
// Layout: x is [CH, SP] row-major (channel-major per IOSurface convention).
// For each spatial position s, we need mean of x[c, s]^2 over c.
//
// Strategy:
//   Pass 1 (streaming): walk channel-blocks of size BLK. For each block,
//   accumulate partial sum-of-squares per spatial position into ss[SP].
//   When done, finalize: ss /= CH, then rrms = 1/sqrt(ss + eps).
//
//   Pass 2 (streaming): walk channel-blocks again. For each block, multiply
//   x[c, :] elementwise by rrms[:], then by gamma[c] scalar per channel.
//
// Compared to the naive in AneRMSNormSmoke:
//   - Removes the big sq buffer entirely (computed inline in registers)
//   - Removes the strided vDSP_sve (we accumulate per-row in contiguous order)
//   - Two streaming passes vs four full-tensor passes
import Foundation
import Accelerate

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

// Naive — matches AneRMSNormSmoke.
fileprivate func rmsNormNaive(_ x: [Float], gamma: [Float], ch: Int, sp: Int, eps: Float,
                              out: inout [Float]) {
    let invCh = Float(1.0) / Float(ch)
    var sq = [Float](repeating: 0, count: ch * sp)
    vDSP_vsq(x, 1, &sq, 1, vDSP_Length(ch * sp))
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
        x.withUnsafeBufferPointer { xb in
            rrms.withUnsafeBufferPointer { rb in
                out.withUnsafeMutableBufferPointer { ob in
                    vDSP_vmul(xb.baseAddress! + c*sp, 1, rb.baseAddress!, 1,
                              ob.baseAddress! + c*sp, 1, vDSP_Length(sp))
                    var gMut = g
                    vDSP_vsmul(ob.baseAddress! + c*sp, 1, &gMut,
                               ob.baseAddress! + c*sp, 1, vDSP_Length(sp))
                }
            }
        }
    }
}

// Streaming version. SP=16 accumulators per pass.
fileprivate func rmsNormStreaming(_ x: [Float], gamma: [Float], ch: Int, sp: Int, eps: Float,
                                  out: inout [Float]) {
    let invCh = Float(1.0) / Float(ch)
    var ss = [Float](repeating: 0, count: sp)

    // Pass 1: stream x once, accumulate per-position sum-of-squares.
    // x[c, s] lives at x[c * sp + s]. Walking c outermost (cache-friendly).
    x.withUnsafeBufferPointer { xb in
        for c in 0..<ch {
            let rowOff = c * sp
            // For each s in 0..<sp: ss[s] += x[c*sp+s]^2
            // vDSP_vsq + vDSP_vadd: square the row into a tmp, add into ss.
            // BUT sp=16 is tiny — Swift compiles a tight loop with SIMD
            // better than two vDSP calls with their setup cost. Inline it.
            for s in 0..<sp {
                let v = xb[rowOff + s]
                ss[s] += v * v
            }
        }
    }

    // Finalize: rrms = 1/sqrt(ss/ch + eps)
    for s in 0..<sp {
        ss[s] = ss[s] * invCh + eps
    }
    var rrms = [Float](repeating: 0, count: sp)
    var nInt: Int32 = Int32(sp)
    vvrsqrtf(&rrms, ss, &nInt)

    // Pass 2: stream x, write y = x * rrms * gamma. Per-channel: y[c, :] =
    // x[c, :] * rrms[:] * gamma[c].
    // For small sp=16 the inner loop autovectorizes well; multiplying by
    // (rrms[s] * gamma[c]) needs a per-channel scalar so we precompute on the fly.
    x.withUnsafeBufferPointer { xb in
        out.withUnsafeMutableBufferPointer { ob in
            for c in 0..<ch {
                let g = gamma[c]
                let rowOff = c * sp
                for s in 0..<sp {
                    ob[rowOff + s] = xb[rowOff + s] * rrms[s] * g
                }
            }
        }
    }
}

fileprivate func cosSim(_ a: [Float], _ b: [Float]) -> Float {
    var d: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0..<a.count { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    return d / (sqrt(na)*sqrt(nb) + 1e-12)
}

@main
struct CpuRMSNormBench {
    static func main() {
        let CH = 1024, SP = 16, EPS: Float = 1e-6
        slog("[rms_bench] CH=\(CH) SP=\(SP)")

        var gamma = randomArray(count: CH, seed: 0xCAFE_FEED, scale: 0.1)
        for i in 0..<CH { gamma[i] += 1.0 }
        let x = randomArray(count: CH * SP, seed: 0xDEAD_BEEF, scale: 0.5)
        var yN = [Float](repeating: 0, count: CH * SP)
        var yS = [Float](repeating: 0, count: CH * SP)

        rmsNormNaive(x, gamma: gamma, ch: CH, sp: SP, eps: EPS, out: &yN)
        rmsNormStreaming(x, gamma: gamma, ch: CH, sp: SP, eps: EPS, out: &yS)
        slog("[rms_bench] cosine vs naive: \(cosSim(yS, yN))")
        var maxErr: Float = 0
        for i in 0..<yN.count {
            let e = abs(yS[i] - yN[i])
            if e > maxErr { maxErr = e }
        }
        slog("[rms_bench] max abs err: \(maxErr)")

        let ITERS = 500
        for _ in 0..<20 { rmsNormNaive(x, gamma: gamma, ch: CH, sp: SP, eps: EPS, out: &yN) }
        let t0 = Date()
        for _ in 0..<ITERS { rmsNormNaive(x, gamma: gamma, ch: CH, sp: SP, eps: EPS, out: &yN) }
        let perNaive = -t0.timeIntervalSinceNow * 1000.0 / Double(ITERS)

        for _ in 0..<20 { rmsNormStreaming(x, gamma: gamma, ch: CH, sp: SP, eps: EPS, out: &yS) }
        let t1 = Date()
        for _ in 0..<ITERS { rmsNormStreaming(x, gamma: gamma, ch: CH, sp: SP, eps: EPS, out: &yS) }
        let perStream = -t1.timeIntervalSinceNow * 1000.0 / Double(ITERS)

        slog("[rms_bench] naive (4-pass vDSP):  \(String(format: "%.4f ms/call", perNaive))")
        slog("[rms_bench] streaming (2-pass):   \(String(format: "%.4f ms/call", perStream))")
        slog("[rms_bench] speedup: \(String(format: "%.2fx", perNaive/perStream))")
        slog("[rms_bench] per-fwd savings across 5 sites: \(String(format: "%.3f ms", 5.0 * (perNaive - perStream)))")
    }
}
