// Single input, three weights via BLOBFILE offsets into one concat blob.
// y = W_a @ x + W_b @ x + W_c @ x
// If this passes: bug is multi-input. If it fails: bug is multi-weight via offsets.
import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(inCh: Int, outCh: Int, sp: Int) throws -> String {
    let url = URL(fileURLWithPath: "donkey-cli/Sources/AneTriWeightSmoke/tri_weight.mil.template")
    let tmpl = try String(contentsOf: url, encoding: .utf8)
    let chunkStride = 64 + outCh * inCh * 2
    return tmpl
        .replacingOccurrences(of: "{IN_CH}", with: String(inCh))
        .replacingOccurrences(of: "{OUT_CH}", with: String(outCh))
        .replacingOccurrences(of: "{SP}", with: String(sp))
        .replacingOccurrences(of: "{OFF_B}", with: String(64 + chunkStride))
        .replacingOccurrences(of: "{OFF_C}", with: String(64 + 2 * chunkStride))
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

fileprivate func buildConcatBlobFP16(weights: [[Float]], rows: Int, cols: Int) -> Data {
    let wsize = rows * cols * 2
    let cs    = 64 + wsize
    let total = 64 + weights.count * cs
    var buf = [UInt8](repeating: 0, count: total)
    buf[0] = 0x01; buf[4] = 0x02
    for (w, weightArr) in weights.enumerated() {
        let chunkOff = 64 + w * cs
        buf[chunkOff + 0] = 0xEF; buf[chunkOff + 1] = 0xBE
        buf[chunkOff + 2] = 0xAD; buf[chunkOff + 3] = 0xDE
        buf[chunkOff + 4] = 0x01
        let ws = UInt32(wsize)
        buf[chunkOff + 8]  = UInt8(ws        & 0xFF)
        buf[chunkOff + 9]  = UInt8((ws >>  8) & 0xFF)
        buf[chunkOff + 10] = UInt8((ws >> 16) & 0xFF)
        buf[chunkOff + 11] = UInt8((ws >> 24) & 0xFF)
        let off = UInt32(chunkOff + 64)
        buf[chunkOff + 16] = UInt8(off        & 0xFF)
        buf[chunkOff + 17] = UInt8((off >>  8) & 0xFF)
        buf[chunkOff + 18] = UInt8((off >> 16) & 0xFF)
        buf[chunkOff + 19] = UInt8((off >> 24) & 0xFF)
        let dataOff = chunkOff + 64
        for i in 0..<(rows * cols) {
            let v = Float16(weightArr[i])
            let bits = v.bitPattern
            buf[dataOff + i*2 + 0] = UInt8(bits & 0xFF)
            buf[dataOff + i*2 + 1] = UInt8((bits >> 8) & 0xFF)
        }
    }
    return Data(buf)
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

@main
struct AneTriWeightSmoke {
    static func main() {
        let IN_CH = 4096, OUT_CH = 1024, SP = 64
        slog("[tri] IN_CH=\(IN_CH) OUT_CH=\(OUT_CH) SP=\(SP) -- 1 input, 3 weights via offsets")

        do { try aneBridgeInit() } catch { slog("init FAIL: \(error)"); exit(1) }

        let W_a = randomArray(count: OUT_CH * IN_CH, seed: 0x1111_1111)
        let W_b = randomArray(count: OUT_CH * IN_CH, seed: 0x2222_2222)
        let W_c = randomArray(count: OUT_CH * IN_CH, seed: 0x3333_3333)
        let blob = buildConcatBlobFP16(weights: [W_a, W_b, W_c], rows: OUT_CH, cols: IN_CH)
        let cs = 64 + OUT_CH * IN_CH * 2
        slog("[tri] blob: \(blob.count) bytes; offsets W_a=64, W_b=\(64+cs), W_c=\(64+2*cs)")

        let mil: String
        do { mil = try loadMIL(inCh: IN_CH, outCh: OUT_CH, sp: SP) } catch {
            slog("template FAIL: \(error)"); exit(2)
        }

        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [("@model_path/weights/weight.bin", blob)],
                inputBytes:  [IN_CH * SP * 4],
                outputBytes: [OUT_CH * SP * 4]
            )
        } catch { slog("compile FAIL: \(error)"); exit(3) }
        slog("[tri] compiled")

        let x = randomArray(count: IN_CH * SP, seed: 0xAAAA_AAAA)
        do {
            try x.withUnsafeBytes { try kernel.writeInput(0, $0.baseAddress!, bytes: $0.count) }
            try kernel.eval()
        } catch { slog("eval FAIL: \(error)"); exit(4) }

        var out = [Float](repeating: 0, count: OUT_CH * SP)
        do {
            try out.withUnsafeMutableBytes { try kernel.readOutput(0, into: $0.baseAddress!, bytes: $0.count) }
        } catch { slog("read FAIL: \(error)"); exit(5) }

        let ya = matmulRef(W: W_a, x: x, outCh: OUT_CH, inCh: IN_CH, sp: SP)
        let yb = matmulRef(W: W_b, x: x, outCh: OUT_CH, inCh: IN_CH, sp: SP)
        let yc = matmulRef(W: W_c, x: x, outCh: OUT_CH, inCh: IN_CH, sp: SP)
        var expected = ya
        for i in 0..<expected.count { expected[i] += yb[i] + yc[i] }

        func cosSim(_ a: [Float], _ b: [Float]) -> Float {
            var d: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0..<a.count { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
            return d / (sqrt(na)*sqrt(nb) + 1e-12)
        }
        let cosFull = cosSim(out, expected)
        let cosA = cosSim(out, ya)
        let cosB = cosSim(out, yb)
        let cosC = cosSim(out, yc)
        slog("[tri] out[0..4]:      \(Array(out[0..<4]))")
        slog("[tri] expected[0..4]: \(Array(expected[0..<4]))")
        slog("[tri] cos vs full sum (W_a+W_b+W_c)@x: \(cosFull)")
        slog("[tri] cos vs W_a@x only: \(cosA)")
        slog("[tri] cos vs W_b@x only: \(cosB)")
        slog("[tri] cos vs W_c@x only: \(cosC)")
        if cosFull < 0.999 { slog("[tri] FAIL"); exit(6) }
        slog("[tri] OK")
    }
}
