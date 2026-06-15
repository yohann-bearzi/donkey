// DonkeyOps — CPU ops used by the donkey forward orchestrator.
//
// All ops are non-throwing, allocation-free in steady state (a few of them
// keep small per-call scratch buffers via inout temporaries the caller owns),
// and operate on UnsafePointer<Float> / UnsafeMutablePointer<Float> so they
// can be called against array, buffer, or IOSurface-mapped memory without
// adapters.
//
// Layout convention: row-major [CH, SP] for hidden states; channel is the
// major axis (matches the ANE IOSurface NCHW with N=1, H=1). Spatial axis
// holds donkey draft positions (K_PAD=16 in v1).
//
// Implementations are the ones validated in donkey-cli/Sources/Cpu*Smoke
// and Cpu*Bench targets. Numerical results documented there.
import Foundation
import Accelerate

// MARK: - RMSNorm (streaming, 3.3x faster than 4-pass vDSP at CH=1024 SP=16).
//
// y[c, s] = x[c, s] * rrms[s] * gamma[c]
//   where ss[s]  = (1/CH) * sum_c x[c, s]^2 + eps
//         rrms[s] = 1 / sqrt(ss[s])
//
// Two streaming passes:
//   Pass 1: walk x once, accumulate sum-of-squares per spatial position into
//           ss[SP]. Scratch buffer (ss) is tiny (SP floats) and stays in L1.
//   Pass 2: walk x again, write y = x * rrms * gamma per channel.
//
// scratch.count must be >= sp. Pass any small reusable [Float] as scratch
// to avoid per-call allocation in hot paths.
public func cpu_rmsnorm(x: UnsafePointer<Float>,
                        gamma: UnsafePointer<Float>,
                        out: UnsafeMutablePointer<Float>,
                        ch: Int, sp: Int, eps: Float,
                        scratch: inout [Float]) {
    precondition(scratch.count >= sp, "rmsnorm scratch must hold sp floats")
    let invCh = Float(1.0) / Float(ch)

    // Pass 1: accumulate sum-of-squares per spatial position.
    // SIMD4 vectorized: process 4 spatial positions per inner-loop iteration.
    // At SP=16 this is 4 SIMD4 ops per channel x 1024 channels = 4K fma vs
    // 16K scalar fma. All ANE-valid SP values (16, 32, 64, 128) are multiples
    // of 4 so the scalar tail is never exercised in practice.
    for s in 0..<sp { scratch[s] = 0 }
    let spVec = sp & ~3  // largest multiple of 4 <= sp
    for c in 0..<ch {
        let rowOff = c * sp
        var s = 0
        scratch.withUnsafeMutableBufferPointer { scratchBuf in
            let sPtr = scratchBuf.baseAddress!
            while s < spVec {
                // Load 4 from x, 4 from scratch, fma into scratch.
                let v = SIMD4<Float>(x[rowOff + s], x[rowOff + s+1],
                                     x[rowOff + s+2], x[rowOff + s+3])
                let acc = SIMD4<Float>(sPtr[s], sPtr[s+1], sPtr[s+2], sPtr[s+3])
                let nxt = acc + v * v
                sPtr[s]   = nxt[0]
                sPtr[s+1] = nxt[1]
                sPtr[s+2] = nxt[2]
                sPtr[s+3] = nxt[3]
                s += 4
            }
        }
        // Scalar tail for sp not divisible by 4.
        while s < sp {
            let v = x[rowOff + s]
            scratch[s] += v * v
            s += 1
        }
    }
    for s in 0..<sp { scratch[s] = scratch[s] * invCh + eps }

    // rrms = 1 / sqrt(scratch). In-place via vvrsqrtf.
    var nInt: Int32 = Int32(sp)
    scratch.withUnsafeMutableBufferPointer { sb in
        vvrsqrtf(sb.baseAddress!, sb.baseAddress!, &nInt)
    }

    // Pass 2: y[c, :] = x[c, :] * scratch[:] * gamma[c].
    for c in 0..<ch {
        let g = gamma[c]
        let rowOff = c * sp
        for s in 0..<sp {
            out[rowOff + s] = x[rowOff + s] * scratch[s] * g
        }
    }
}

// MARK: - SiLU (fused vvexpf-based, 0.11 ms at HIDDEN=4096 SP=16).
//
// y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
//
// Strategy: process input in CHUNK-sized blocks. For each chunk:
//   1. scratch = -x (one read of x, one write of scratch — scratch stays in L1)
//   2. scratch = exp(scratch) in place via vvexpf
//   3. y[i] = x[i] / (1 + scratch[i]) in a tight inner loop
//
// Caller supplies scratch buffer (>= CHUNK floats). This avoids per-call alloc.
public func cpu_silu(x: UnsafePointer<Float>,
                     out: UnsafeMutablePointer<Float>,
                     count: Int,
                     scratch: inout [Float]) {
    let CHUNK = 4096
    precondition(scratch.count >= CHUNK, "silu scratch must hold at least \(CHUNK) floats")

    var pos = 0
    while pos < count {
        let n = min(CHUNK, count - pos)
        var nInt: Int32 = Int32(n)
        var negOne: Float = -1.0

        scratch.withUnsafeMutableBufferPointer { sb in
            vDSP_vsmul(x + pos, 1, &negOne, sb.baseAddress!, 1, vDSP_Length(n))
            vvexpf(sb.baseAddress!, sb.baseAddress!, &nInt)
        }
        for i in 0..<n {
            out[pos + i] = x[pos + i] / (1.0 + scratch[i])
        }
        pos += n
    }
}

// MARK: - Residual add (elementwise, no broadcasting).
//
// out = a + b.  Both inputs same shape and layout. Uses vDSP_vadd.
public func cpu_residual_add(a: UnsafePointer<Float>,
                             b: UnsafePointer<Float>,
                             out: UnsafeMutablePointer<Float>,
                             count: Int) {
    vDSP_vadd(a, 1, b, 1, out, 1, vDSP_Length(count))
}

// MARK: - Scalar sigmoid for the confidence head.
//
// Single-element sigmoid: 1 / (1 + exp(-x)). Called per draft position on
// the OUT_CONF=1 channel after the output head. Trivial cost — kept here for
// uniformity with the rest of the CPU op surface.
public func cpu_sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + expf(-x))
}


// MARK: - LayerNorm (channel-axis, mirrors cpu_rmsnorm shape).
//
// y[c, s] = ((x[c, s] - mean[s]) / sqrt(var[s] + eps)) * gamma[c] + beta[c]
//   where mean[s] = (1/CH) * sum_c x[c, s]
//          var[s] = (1/CH) * sum_c (x[c, s] - mean[s])^2
//
// Two streaming passes over x, mean+var accumulated jointly via:
//   var = E[x^2] - (E[x])^2
// This lets pass 1 collect sum and sum-of-squares in a single channel walk,
// matching cpu_rmsnorm's pattern.
//
// scratchMean and scratchVar must each hold >= sp floats.
public func cpu_layernorm(x: UnsafePointer<Float>,
                          gamma: UnsafePointer<Float>,
                          beta: UnsafePointer<Float>,
                          out: UnsafeMutablePointer<Float>,
                          ch: Int, sp: Int, eps: Float,
                          scratchMean: inout [Float],
                          scratchVar: inout [Float]) {
    precondition(scratchMean.count >= sp, "layernorm scratchMean must hold sp floats")
    precondition(scratchVar.count >= sp,  "layernorm scratchVar  must hold sp floats")
    let invCh = Float(1.0) / Float(ch)

    // Pass 1: per-spatial-position accumulators for sum and sum-of-squares.
    // SIMD4 over the 4-aligned head; scalar tail otherwise. Same pattern as
    // cpu_rmsnorm so the access pattern matches.
    for s in 0..<sp { scratchMean[s] = 0; scratchVar[s] = 0 }
    let spVec = sp & ~3
    for c in 0..<ch {
        let rowOff = c * sp
        var s = 0
        scratchMean.withUnsafeMutableBufferPointer { meanBuf in
            scratchVar.withUnsafeMutableBufferPointer { varBuf in
                let mPtr = meanBuf.baseAddress!
                let vPtr = varBuf.baseAddress!
                while s < spVec {
                    let v = SIMD4<Float>(x[rowOff + s], x[rowOff + s+1],
                                         x[rowOff + s+2], x[rowOff + s+3])
                    let mAcc = SIMD4<Float>(mPtr[s], mPtr[s+1], mPtr[s+2], mPtr[s+3])
                    let vAcc = SIMD4<Float>(vPtr[s], vPtr[s+1], vPtr[s+2], vPtr[s+3])
                    let mNxt = mAcc + v
                    let vNxt = vAcc + v * v
                    mPtr[s]   = mNxt[0]; mPtr[s+1] = mNxt[1]
                    mPtr[s+2] = mNxt[2]; mPtr[s+3] = mNxt[3]
                    vPtr[s]   = vNxt[0]; vPtr[s+1] = vNxt[1]
                    vPtr[s+2] = vNxt[2]; vPtr[s+3] = vNxt[3]
                    s += 4
                }
            }
        }
        while s < sp {
            let v = x[rowOff + s]
            scratchMean[s] += v
            scratchVar[s]  += v * v
            s += 1
        }
    }
    // mean[s] = sum/ch ; var[s] = E[x^2] - mean^2 ; invstd = 1/sqrt(var+eps)
    for s in 0..<sp {
        let m = scratchMean[s] * invCh
        let varS = scratchVar[s] * invCh - m * m
        scratchMean[s] = m                  // stash mean
        scratchVar[s]  = varS + eps         // stash variance + eps
    }
    var nInt: Int32 = Int32(sp)
    scratchVar.withUnsafeMutableBufferPointer { vb in
        vvrsqrtf(vb.baseAddress!, vb.baseAddress!, &nInt)   // var -> invstd in place
    }

    // Pass 2: y[c, s] = (x - mean[s]) * invstd[s] * gamma[c] + beta[c]
    for c in 0..<ch {
        let rowOff = c * sp
        let g = gamma[c]
        let b = beta[c]
        var s = 0
        while s < spVec {
            let xv = SIMD4<Float>(x[rowOff + s], x[rowOff + s+1],
                                  x[rowOff + s+2], x[rowOff + s+3])
            let mv = SIMD4<Float>(scratchMean[s], scratchMean[s+1],
                                  scratchMean[s+2], scratchMean[s+3])
            let iv = SIMD4<Float>(scratchVar[s], scratchVar[s+1],
                                  scratchVar[s+2], scratchVar[s+3])
            let yv = (xv - mv) * iv * g + b
            out[rowOff + s]   = yv[0]; out[rowOff + s+1] = yv[1]
            out[rowOff + s+2] = yv[2]; out[rowOff + s+3] = yv[3]
            s += 4
        }
        while s < sp {
            out[rowOff + s] = (x[rowOff + s] - scratchMean[s]) * scratchVar[s] * g + b
            s += 1
        }
    }
}
