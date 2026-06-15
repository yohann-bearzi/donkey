// Donkey kernel #5 — SiLU activation.
//
// y = x * sigmoid(x). Two MIL ops. Used in the FFN between up and down
// projections, on tensors of shape [1, HIDDEN=4096, 1, SP=16].
//
// Latency expectation: should be well under 0.2 ms even at HIDDEN=4096 -
// it's just two elementwise ops over 65k elements.
import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(ch: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-trainer/kernels/silu.mil.template")
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

fileprivate func siluRef(_ x: [Float]) -> [Float] {
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
struct AneSiLUSmoke {
    static func main() {
        let CH = 4096
        let SP = 16
        slog("[silu] CH=\(CH) SP=\(SP)")

        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }

        let mil: String
        do { mil = try loadMIL(ch: CH, sp: SP) } catch {
            slog("template FAIL: \(error)"); exit(2)
        }

        // SiLU has no weights, but aneCompile requires at least one weight entry —
        // we use a dummy 1-byte blob. Actually let's just pass an empty array.
        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [],   // no weights for pure activation
                inputBytes:  [CH * SP * 4],
                outputBytes: [CH * SP * 4]
            )
        } catch { slog("compile FAIL: \(error)"); exit(3) }
        slog("[silu] compiled, compile_count=\(aneCompileCount())")

        // Inputs spanning typical FFN activation range.
        let x = randomArray(count: CH * SP, seed: 0xF00D_BEEF, scale: 2.0)

        do {
            try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            try kernel.eval()
        } catch { slog("eval FAIL: \(error)"); exit(4) }

        var out = [Float](repeating: 0, count: CH * SP)
        do {
            try out.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
        } catch { slog("read FAIL: \(error)"); exit(5) }

        let expected = siluRef(x)
        let cos = cosSim(out, expected)

        var maxAbsErr: Float = 0
        for i in 0..<out.count {
            let e = abs(out[i] - expected[i])
            if e > maxAbsErr { maxAbsErr = e }
        }

        slog("[silu] out[0..4]:      \(Array(out[0..<4]))")
        slog("[silu] expected[0..4]: \(Array(expected[0..<4]))")
        slog("[silu] cosine sim:     \(cos)")
        slog("[silu] max abs err:    \(maxAbsErr)")

        if cos < 0.999 {
            slog("[silu] FAIL: cosine \(cos) < 0.999")
            exit(6)
        }

        // Latency: warmup + 100 iters.
        for _ in 0..<10 {
            do {
                try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
                try kernel.eval()
            } catch { exit(7) }
        }
        let t0 = Date()
        for _ in 0..<100 {
            do {
                try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
                try kernel.eval()
            } catch { exit(8) }
        }
        let perCall = -t0.timeIntervalSinceNow * 1000.0 / 100.0
        slog("[silu] steady-state: \(String(format: "%.3f ms/call", perCall))")
        slog("[silu] OK")
    }
}
