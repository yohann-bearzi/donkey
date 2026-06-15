// OpsTests — Phase 2a of donkey v2 test harness.
//
// For each CPU op (rmsnorm, silu, residual_add, sigmoid):
//   1. Generate seeded deterministic input.
//   2. Write input to <out_dir>/<op>_input.bin (fp32 row-major).
//   3. Run the canonical implementation from DonkeyOps.
//   4. Write Swift output to <out_dir>/<op>_swift.bin.
//
// Also writes edge-case inputs/outputs labeled with suffix "_edge".
//
// A separate Python validator reads these files, computes the naive
// reference, asserts agreement.
//
// usage: ops-tests <out_dir>
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

fileprivate func writeFloats(_ a: [Float], to path: String) throws {
    let data = a.withUnsafeBufferPointer { Data(buffer: $0) }
    try data.write(to: URL(fileURLWithPath: path))
}

// === Canonical shapes (match v1 / v2 default config: hidden=1024, sp=16, ffn=4096) ===
let CH = 1024
let SP = 16
let FFN_N = 4096 * 16

// === Test 1: RMSNorm ===
fileprivate func testRmsnorm(_ outDir: String) throws {
    let x = randomArray(count: CH * SP, seed: 0xDEAD_BEEF, scale: 0.5)
    var gamma = randomArray(count: CH, seed: 0xCAFE_FEED, scale: 0.1)
    for i in 0..<CH { gamma[i] += 1.0 }
    var out = [Float](repeating: 0, count: CH * SP)
    var scratch = [Float](repeating: 0, count: SP)
    x.withUnsafeBufferPointer { xb in
        gamma.withUnsafeBufferPointer { gb in
            out.withUnsafeMutableBufferPointer { ob in
                cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                            out: ob.baseAddress!, ch: CH, sp: SP, eps: 1e-6,
                            scratch: &scratch)
            }
        }
    }
    try writeFloats(x,     to: "\(outDir)/rmsnorm_input.bin")
    try writeFloats(gamma, to: "\(outDir)/rmsnorm_gamma.bin")
    try writeFloats(out,   to: "\(outDir)/rmsnorm_swift.bin")

    // Edge case: row with all-zero values at one spatial position.
    var xEdge = randomArray(count: CH * SP, seed: 0xBEEF_DEAD, scale: 0.5)
    let zeroSp = 5
    for c in 0..<CH { xEdge[c * SP + zeroSp] = 0 }
    var outEdge = [Float](repeating: 0, count: CH * SP)
    xEdge.withUnsafeBufferPointer { xb in
        gamma.withUnsafeBufferPointer { gb in
            outEdge.withUnsafeMutableBufferPointer { ob in
                cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                            out: ob.baseAddress!, ch: CH, sp: SP, eps: 1e-6,
                            scratch: &scratch)
            }
        }
    }
    try writeFloats(xEdge,   to: "\(outDir)/rmsnorm_edge_input.bin")
    try writeFloats(outEdge, to: "\(outDir)/rmsnorm_edge_swift.bin")
    slog("[ops] rmsnorm: input + output + edge written")
}

// === Test 2: SiLU ===
fileprivate func testSilu(_ outDir: String) throws {
    let x = randomArray(count: FFN_N, seed: 0xF00D_BEEF, scale: 2.0)
    var out = [Float](repeating: 0, count: FFN_N)
    var scratch = [Float](repeating: 0, count: 4096)
    x.withUnsafeBufferPointer { xb in
        out.withUnsafeMutableBufferPointer { ob in
            cpu_silu(x: xb.baseAddress!, out: ob.baseAddress!,
                     count: FFN_N, scratch: &scratch)
        }
    }
    try writeFloats(x,   to: "\(outDir)/silu_input.bin")
    try writeFloats(out, to: "\(outDir)/silu_swift.bin")

    // Edge case: saturation. Inputs include +20, -20, 0 to test both clamp paths.
    var xEdge: [Float] = (0..<64).map { i in
        switch i % 4 {
        case 0:  return 20.0   // saturate positive: silu(x) ≈ x
        case 1:  return -20.0  // saturate negative: silu(x) ≈ 0
        case 2:  return 0.0    // silu(0) = 0
        default: return Float(i) * 0.1  // smooth values
        }
    }
    let nEdge = xEdge.count
    var outEdge = [Float](repeating: 0, count: nEdge)
    var scratch2 = [Float](repeating: 0, count: 4096)
    xEdge.withUnsafeBufferPointer { xb in
        outEdge.withUnsafeMutableBufferPointer { ob in
            cpu_silu(x: xb.baseAddress!, out: ob.baseAddress!,
                     count: nEdge, scratch: &scratch2)
        }
    }
    try writeFloats(xEdge,   to: "\(outDir)/silu_edge_input.bin")
    try writeFloats(outEdge, to: "\(outDir)/silu_edge_swift.bin")
    _ = xEdge  // suppress unused-var warning
    slog("[ops] silu: input + output + edge written")
}

// === Test 3: Residual add ===
fileprivate func testResidualAdd(_ outDir: String) throws {
    let a = randomArray(count: CH * SP, seed: 0xAAAA_0001, scale: 0.5)
    let b = randomArray(count: CH * SP, seed: 0xBBBB_0002, scale: 0.5)
    var out = [Float](repeating: 0, count: CH * SP)
    a.withUnsafeBufferPointer { ab in
        b.withUnsafeBufferPointer { bb in
            out.withUnsafeMutableBufferPointer { ob in
                cpu_residual_add(a: ab.baseAddress!, b: bb.baseAddress!,
                                 out: ob.baseAddress!, count: CH * SP)
            }
        }
    }
    try writeFloats(a,   to: "\(outDir)/add_a.bin")
    try writeFloats(b,   to: "\(outDir)/add_b.bin")
    try writeFloats(out, to: "\(outDir)/add_swift.bin")

    // Edge case: in-place add (a and out aliased). Result must equal a + b.
    var inplace = a  // copy
    inplace.withUnsafeMutableBufferPointer { ab in
        b.withUnsafeBufferPointer { bb in
            cpu_residual_add(a: ab.baseAddress!, b: bb.baseAddress!,
                             out: ab.baseAddress!, count: CH * SP)
        }
    }
    try writeFloats(inplace, to: "\(outDir)/add_inplace_swift.bin")
    slog("[ops] residual_add: a + b + output + inplace written")
}

// === Test 4: Sigmoid (scalar) ===
fileprivate func testSigmoid(_ outDir: String) throws {
    // Range covers extremes: -1000, -10, -1, 0, 1, 10, 1000.
    let xs: [Float] = [-1000, -100, -10, -1, -0.5, 0, 0.5, 1, 10, 100, 1000]
    var ys = [Float](repeating: 0, count: xs.count)
    for i in 0..<xs.count { ys[i] = cpu_sigmoid(xs[i]) }
    try writeFloats(xs, to: "\(outDir)/sigmoid_input.bin")
    try writeFloats(ys, to: "\(outDir)/sigmoid_swift.bin")
    slog("[ops] sigmoid: input + output written")
}

// === Performance regression check (informational; pass/fail in Python). ===
fileprivate func benchOps(_ outDir: String) throws {
    let ITERS = 200
    let x = randomArray(count: CH * SP, seed: 0xBEEF, scale: 0.5)
    var gamma = [Float](repeating: 1.0, count: CH)
    var out = [Float](repeating: 0, count: CH * SP)
    var scratch = [Float](repeating: 0, count: SP)
    let xs = randomArray(count: FFN_N, seed: 0xDEAD, scale: 2.0)
    var outs = [Float](repeating: 0, count: FFN_N)
    var sscratch = [Float](repeating: 0, count: 4096)

    // Warm up. First-run penalty on a cold binary (CPU freq scaling, page
    // faults, branch predictor) is substantial; 200 iters gets us to steady
    // state reliably across runs.
    for _ in 0..<200 {
        x.withUnsafeBufferPointer { xb in gamma.withUnsafeBufferPointer { gb in
            out.withUnsafeMutableBufferPointer { ob in
                cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                            out: ob.baseAddress!, ch: CH, sp: SP, eps: 1e-6,
                            scratch: &scratch)
            }}}
    }
    // Time 3 times, keep last. First measurement absorbs process-startup
    // costs (CPU freq scaling, page-in) that aren't representative of the
    // steady-state per-op cost we want to track.
    var dRms: Double = 0
    for _ in 0..<3 {
        let t0 = Date()
        for _ in 0..<ITERS {
            x.withUnsafeBufferPointer { xb in gamma.withUnsafeBufferPointer { gb in
                out.withUnsafeMutableBufferPointer { ob in
                    cpu_rmsnorm(x: xb.baseAddress!, gamma: gb.baseAddress!,
                                out: ob.baseAddress!, ch: CH, sp: SP, eps: 1e-6,
                                scratch: &scratch)
                }}}
        }
        dRms = -t0.timeIntervalSinceNow * 1000.0 / Double(ITERS)
    }

    for _ in 0..<200 {
        xs.withUnsafeBufferPointer { xb in
            outs.withUnsafeMutableBufferPointer { ob in
                cpu_silu(x: xb.baseAddress!, out: ob.baseAddress!,
                         count: FFN_N, scratch: &sscratch)
            }
        }
    }
    var dSilu: Double = 0
    for _ in 0..<3 {
        let t1 = Date()
        for _ in 0..<ITERS {
            xs.withUnsafeBufferPointer { xb in
                outs.withUnsafeMutableBufferPointer { ob in
                    cpu_silu(x: xb.baseAddress!, out: ob.baseAddress!,
                             count: FFN_N, scratch: &sscratch)
                }
            }
        }
        dSilu = -t1.timeIntervalSinceNow * 1000.0 / Double(ITERS)
    }

    // Write benchmark results as JSON for Python to compare against thresholds.
    let json = """
    {
        "rmsnorm_ms": \(dRms),
        "silu_ms": \(dSilu),
        "rmsnorm_threshold_ms": 0.040,
        "silu_threshold_ms": 0.080
    }
    """
    try json.write(toFile: "\(outDir)/perf.json", atomically: true, encoding: .utf8)
    slog("[ops] perf: rmsnorm=\(String(format: "%.4f", dRms))ms silu=\(String(format: "%.4f", dSilu))ms")
}

@main
struct OpsTests {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 2 else {
            slog("usage: ops-tests <out_dir>")
            exit(1)
        }
        let outDir = args[1]
        try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

        try testRmsnorm(outDir)
        try testSilu(outDir)
        try testResidualAdd(outDir)
        try testSigmoid(outDir)
        try benchOps(outDir)
        slog("[ops] all ops + edge cases written to \(outDir)")
    }
}
