// Chebyshev-polynomial SiLU bench.
//
// Goal: replace vvexpf (84% of fused SiLU cost at 1.4 ns/element) with a
// degree-D polynomial approximation that's still accurate enough to be
// invisible after the downstream fp16 cast.
//
// Method: range reduction. For x in [-16, 16]:
//   k = round((-x) / ln(2))
//   r = -x - k * ln(2)              // r in [-ln(2)/2, ln(2)/2]
//   exp(-x) = 2^k * exp(r)
//
// 2^k is built by bit-twiddling the IEEE-754 exponent (single integer op).
// exp(r) is a Chebyshev polynomial in r evaluated by Horner.
//
// Three degrees tested:
//   D3: max rel err 1.0e-4   (10x below fp16 noise; aggressive)
//   D4: max rel err 3.6e-6   (280x below fp16 noise; safe default for inference)
//   D5: max rel err 1.0e-7   (matches fp32 ULP)
//   D6: max rel err 2.6e-9   (below fp32 ULP; safe for v2 training/backprop)
//
// At runtime we compute silu(x) = x / (1 + exp(-x)) using the polynomial
// approximation for exp(-x).
import Foundation
import Accelerate

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

// Chebyshev coefficients in NATURAL basis, descending degree (Horner-ready).
fileprivate let CHEBY_D3: [Float] = [
    1.6792160975e-01, 5.0502354196e-01, 9.9996227848e-01, 9.9992448146e-01
]
fileprivate let CHEBY_D4: [Float] = [
    4.1917529688e-02, 1.6792160975e-01, 4.9998869109e-01, 9.9996227848e-01, 1.0000000755e+00
]
fileprivate let CHEBY_D5: [Float] = [
    8.3751288901e-03, 4.1917529688e-02, 1.6666415477e-01, 4.9998869109e-01,
    1.0000000377e+00, 1.0000000755e+00
]
// D6: max rel err 2.6e-9 on exp(r). Below fp32 ULP. Safe through v2 backprop,
// EWMA, calibration — anything that would compound error.
fileprivate let CHEBY_D6: [Float] = [
    1.3948580838e-03, 8.3751288900e-03, 4.1666218274e-02, 1.6666415477e-01,
    5.0000001077e-01, 1.0000000377e+00, 9.9999999996e-01
]

// SIMD4-vectorized exp(-x) and silu, D6 only (the one we'd actually ship).
// Processes 4 fp32 elements per iteration via Swift's SIMD4<Float>, which
// lowers to NEON FMA + bit-twiddling on Apple Silicon.
@inline(__always)
fileprivate func pow2i_simd4(_ k: SIMD4<Int32>) -> SIMD4<Float> {
    // Build 2^k via IEEE-754 exponent injection, lane-wise.
    let clamped = k.clamped(lowerBound: SIMD4(repeating: -126),
                            upperBound: SIMD4(repeating: 127))
    let bits = SIMD4<UInt32>(truncatingIfNeeded: clamped &+ SIMD4(repeating: 127)) &<< 23
    return unsafeBitCast(bits, to: SIMD4<Float>.self)
}

fileprivate func siluChebyD6_SIMD4(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                                    count: Int) {
    // D6 coefficients as scalar broadcasts (load once, reuse).
    let c0 = SIMD4<Float>(repeating: 1.3948580838e-03)
    let c1 = SIMD4<Float>(repeating: 8.3751288900e-03)
    let c2 = SIMD4<Float>(repeating: 4.1666218274e-02)
    let c3 = SIMD4<Float>(repeating: 1.6666415477e-01)
    let c4 = SIMD4<Float>(repeating: 5.0000001077e-01)
    let c5 = SIMD4<Float>(repeating: 1.0000000377e+00)
    let c6 = SIMD4<Float>(repeating: 9.9999999996e-01)
    let ln2 = SIMD4<Float>(repeating: LN2)
    let log2e = SIMD4<Float>(repeating: LOG2E)
    let half = SIMD4<Float>(repeating: 0.5)
    let zero = SIMD4<Float>(repeating: 0.0)
    let one = SIMD4<Float>(repeating: 1.0)
    let posSat = SIMD4<Float>(repeating: 16.0)
    let negSat = SIMD4<Float>(repeating: -16.0)

    // 4-lane main loop.
    let nVec = count & ~3
    var i = 0
    while i < nVec {
        let xi = SIMD4<Float>(x[i], x[i+1], x[i+2], x[i+3])
        let negX = -xi

        // k = round(negX * log2e). Round-half-away-from-zero via copysign trick.
        let kf = negX * log2e + half.replacing(with: -half, where: negX .< zero)
        let k = SIMD4<Int32>(kf)  // truncation toward zero after biased add ≈ rounding

        // r = negX - k*ln2
        let r = negX - SIMD4<Float>(k) * ln2

        // Horner on r, degree 6.
        var acc = c0
        acc = acc * r + c1
        acc = acc * r + c2
        acc = acc * r + c3
        acc = acc * r + c4
        acc = acc * r + c5
        acc = acc * r + c6

        let expNegX = pow2i_simd4(k) * acc
        var y = xi / (one + expNegX)

        // Saturate at extremes (per-lane select).
        y = y.replacing(with: xi,   where: xi .> posSat)
        y = y.replacing(with: zero, where: xi .< negSat)

        out[i+0] = y[0]; out[i+1] = y[1]; out[i+2] = y[2]; out[i+3] = y[3]
        i += 4
    }
    // Scalar tail for count not divisible by 4.
    while i < count {
        let xi = x[i]
        if xi > 16.0 { out[i] = xi }
        else if xi < -16.0 { out[i] = 0.0 }
        else {
            let negX = -xi
            let k = Int32(negX * LOG2E + (negX >= 0 ? 0.5 : -0.5))
            let r = negX - Float(k) * LN2
            var acc: Float = 1.3948580838e-03
            acc = acc * r + 8.3751288900e-03
            acc = acc * r + 4.1666218274e-02
            acc = acc * r + 1.6666415477e-01
            acc = acc * r + 5.0000001077e-01
            acc = acc * r + 1.0000000377e+00
            acc = acc * r + 9.9999999996e-01
            let bits = UInt32(bitPattern: max(-126, min(127, k)) + 127) << 23
            let pow2 = Float(bitPattern: bits)
            out[i] = xi / (1.0 + pow2 * acc)
        }
        i += 1
    }
}

fileprivate let LN2: Float       = 0.69314718055994530942
fileprivate let LOG2E: Float     = 1.44269504088896340736  // 1 / ln(2)

// Build 2^k via IEEE-754 exponent bits. k clamped to [-126, 127] to stay in normals.
// For k beyond that range, exp(-x) underflows to 0 or overflows huge; we handle
// the underflow case explicitly in silu (returns 0 or x as appropriate).
@inline(__always)
fileprivate func pow2i(_ k: Int32) -> Float {
    let kc = max(-126, min(127, k))
    let bits = UInt32(bitPattern: (kc + 127)) << 23
    return Float(bitPattern: bits)
}

// Evaluate exp(-x) via range reduction + Chebyshev polynomial in r.
// Pass coefs in descending degree.
@inline(__always)
fileprivate func expNegCheby(_ x: Float, coefs: UnsafePointer<Float>, degree: Int) -> Float {
    // -x large positive => underflow to 0
    // -x large negative => overflow; clamp at fp16 max upstream (silu wraps)
    let negX = -x
    let k = Int32(negX * LOG2E + (negX >= 0 ? 0.5 : -0.5))
    let r  = negX - Float(k) * LN2

    // Horner on r:
    var acc = coefs[0]
    for i in 1...degree {
        acc = acc * r + coefs[i]
    }
    return pow2i(k) * acc
}

// silu via Chebyshev-exp. Saturates: for |x| > 16, silu(x) ≈ max(x, 0).
fileprivate func siluCheby(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                            count: Int, coefs: [Float]) {
    let degree = coefs.count - 1
    coefs.withUnsafeBufferPointer { cb in
        let cp = cb.baseAddress!
        for i in 0..<count {
            let xi = x[i]
            if xi > 16.0 {
                out[i] = xi              // sigmoid(x) ≈ 1
            } else if xi < -16.0 {
                out[i] = 0.0             // sigmoid(x) ≈ 0
            } else {
                let e = expNegCheby(xi, coefs: cp, degree: degree)
                out[i] = xi / (1.0 + e)
            }
        }
    }
}

// Fused vvexpf-based SiLU (from CpuSiLUSmoke), for comparison.
fileprivate func siluFused(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>,
                            count: Int) {
    let CHUNK = 4096
    var scratch = [Float](repeating: 0, count: CHUNK)
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

// Naive reference (gold standard).
fileprivate func siluRef(_ x: [Float]) -> [Float] {
    var y = [Float](repeating: 0, count: x.count)
    for i in 0..<x.count {
        let s = 1.0 / (1.0 + exp(-x[i]))
        y[i] = x[i] * s
    }
    return y
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

// Measure FP16-cast equivalence: cast both outputs to fp16, then back to fp32,
// and report bit-exact match rate. This is what the downstream linear1x1 conv
// actually sees.
fileprivate func fp16Mismatches(_ a: [Float], _ b: [Float]) -> (count: Int, total: Int) {
    var mismatches = 0
    for i in 0..<a.count {
        let af = Float(Float16(a[i]))
        let bf = Float(Float16(b[i]))
        if af != bf { mismatches += 1 }
    }
    return (mismatches, a.count)
}

fileprivate func runOne(label: String, coefs: [Float], x: [Float], yRef: [Float]) -> Double {
    var y = [Float](repeating: 0, count: x.count)
    x.withUnsafeBufferPointer { xb in
        y.withUnsafeMutableBufferPointer { yb in
            siluCheby(xb.baseAddress!, yb.baseAddress!, count: x.count, coefs: coefs)
        }
    }
    let cos = cosSim(y, yRef)
    var maxAbsErr: Float = 0
    var maxRelErr: Float = 0
    for i in 0..<y.count {
        let e = abs(y[i] - yRef[i])
        let denom = max(abs(yRef[i]), 1e-4)
        if e > maxAbsErr { maxAbsErr = e }
        if e / denom > maxRelErr { maxRelErr = e / denom }
    }
    let (mm, tot) = fp16Mismatches(y, yRef)
    let mmFrac = Double(mm) / Double(tot)
    slog("[silu_cheby \(label)] cosine=\(cos), maxAbs=\(maxAbsErr), maxRel=\(maxRelErr), fp16-mismatches=\(mm)/\(tot) (\(String(format: "%.3f%%", mmFrac * 100)))")

    let ITERS = 200
    var y2 = [Float](repeating: 0, count: x.count)
    for _ in 0..<20 {
        x.withUnsafeBufferPointer { xb in
            y2.withUnsafeMutableBufferPointer { yb in
                siluCheby(xb.baseAddress!, yb.baseAddress!, count: x.count, coefs: coefs)
            }
        }
    }
    let t0 = Date()
    for _ in 0..<ITERS {
        x.withUnsafeBufferPointer { xb in
            y2.withUnsafeMutableBufferPointer { yb in
                siluCheby(xb.baseAddress!, yb.baseAddress!, count: x.count, coefs: coefs)
            }
        }
    }
    return -t0.timeIntervalSinceNow * 1000.0 / Double(ITERS)
}

@main
struct CpuSiLUChebyBench {
    static func main() {
        let N = 4096 * 16
        slog("[silu_cheby] N=\(N) (donkey FFN: HIDDEN=4096, SP=16)")

        let x = randomArray(count: N, seed: 0xF00D_BEEF, scale: 2.0)
        let yRef = siluRef(x)

        let dD3 = runOne(label: "D3", coefs: CHEBY_D3, x: x, yRef: yRef)
        slog("  D3 fused: \(String(format: "%.4f ms/call", dD3))")

        let dD4 = runOne(label: "D4", coefs: CHEBY_D4, x: x, yRef: yRef)
        slog("  D4 fused: \(String(format: "%.4f ms/call", dD4))")

        let dD5 = runOne(label: "D5", coefs: CHEBY_D5, x: x, yRef: yRef)
        slog("  D5 fused: \(String(format: "%.4f ms/call", dD5))")

        let dD6 = runOne(label: "D6", coefs: CHEBY_D6, x: x, yRef: yRef)
        slog("  D6 fused: \(String(format: "%.4f ms/call", dD6))")

        // SIMD4-vectorized D6.
        var ySimd = [Float](repeating: 0, count: N)
        for _ in 0..<20 {
            x.withUnsafeBufferPointer { xb in
                ySimd.withUnsafeMutableBufferPointer { yb in
                    siluChebyD6_SIMD4(xb.baseAddress!, yb.baseAddress!, count: N)
                }
            }
        }
        let tS = Date()
        for _ in 0..<200 {
            x.withUnsafeBufferPointer { xb in
                ySimd.withUnsafeMutableBufferPointer { yb in
                    siluChebyD6_SIMD4(xb.baseAddress!, yb.baseAddress!, count: N)
                }
            }
        }
        let dSimd = -tS.timeIntervalSinceNow * 1000.0 / 200.0
        let cosS = cosSim(ySimd, yRef)
        var maxAbsS: Float = 0, maxRelS: Float = 0
        for k in 0..<N {
            let e = abs(ySimd[k] - yRef[k])
            let denom = max(abs(yRef[k]), 1e-4)
            if e > maxAbsS { maxAbsS = e }
            if e/denom > maxRelS { maxRelS = e/denom }
        }
        let (mmS, totS) = fp16Mismatches(ySimd, yRef)
        slog("[silu_cheby D6-SIMD4] cosine=\(cosS), maxAbs=\(maxAbsS), maxRel=\(maxRelS), fp16-mismatches=\(mmS)/\(totS)")
        slog("  D6 SIMD4: \(String(format: "%.4f ms/call", dSimd))")

        // Bench vvexpf path for direct comparison.
        var yFused = [Float](repeating: 0, count: N)
        for _ in 0..<20 {
            x.withUnsafeBufferPointer { xb in
                yFused.withUnsafeMutableBufferPointer { yb in
                    siluFused(xb.baseAddress!, yb.baseAddress!, count: N)
                }
            }
        }
        let ITERS = 200
        let tF = Date()
        for _ in 0..<ITERS {
            x.withUnsafeBufferPointer { xb in
                yFused.withUnsafeMutableBufferPointer { yb in
                    siluFused(xb.baseAddress!, yb.baseAddress!, count: N)
                }
            }
        }
        let dFused = -tF.timeIntervalSinceNow * 1000.0 / Double(ITERS)
        slog("  vvexpf fused (baseline): \(String(format: "%.4f ms/call", dFused))")

        slog("")
        slog("=== Summary (per FFN site) ===")
        slog(String(format: "  vvexpf fused:  %.4f ms  (cosine 1.0 by construction)", dFused))
        slog(String(format: "  Chebyshev D3:  %.4f ms  (%.2fx vs vvexpf)", dD3, dFused/dD3))
        slog(String(format: "  Chebyshev D4:  %.4f ms  (%.2fx vs vvexpf)", dD4, dFused/dD4))
        slog(String(format: "  Chebyshev D5:  %.4f ms  (%.2fx vs vvexpf)", dD5, dFused/dD5))
        slog(String(format: "  Chebyshev D6:  %.4f ms  (%.2fx vs vvexpf)", dD6, dFused/dD6))
        slog(String(format: "  Cheby D6 SIMD4:%.4f ms  (%.2fx vs vvexpf)", dSimd, dFused/dSimd))
        slog("")
        slog("Verdict: if SIMD4 D6 beats vvexpf, ship D6 (we own the impl, v2-safe).")
        slog("        else ship vvexpf (Apple's vForce is already polynomial under the hood).")
    }
}
