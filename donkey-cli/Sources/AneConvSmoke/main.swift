import Foundation
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func loadMIL(ch: Int, sp: Int) throws -> String {
    let srcPath = "donkey-cli/Sources/AneConvSmoke/conv1x1.mil.template"
    let url = URL(fileURLWithPath: srcPath)
    let tmpl = try String(contentsOf: url, encoding: .utf8)
    return tmpl
        .replacingOccurrences(of: "{CH}", with: String(ch))
        .replacingOccurrences(of: "{SP}", with: String(sp))
}

@main
struct AneConvSmoke {
    static func main() {
        let CH = 256
        let SP = 64
        slog("[conv] CH=\(CH) SP=\(SP) -- 1x1 conv smoke")

        do { try aneBridgeInit() } catch {
            slog("[conv] init FAIL: \(error)"); exit(1)
        }
        slog("[conv] bridge ok")

        var weights = [Float](repeating: 0, count: CH * CH)
        for i in 0..<CH {
            weights[i * CH + i] = 1.0  // identity passthrough
        }
        let weightBlob = aneBuildWeightBlobFP16(weights, rows: CH, cols: CH)
        slog("[conv] weight blob: \(weightBlob.count) bytes (expect \(128 + CH * CH * 2))")

        let mil: String
        do {
            mil = try loadMIL(ch: CH, sp: SP)
        } catch {
            slog("[conv] template load FAIL: \(error)"); exit(2)
        }
        slog("[conv] MIL: \(mil.count) chars")

        let inBytes  = CH * SP * 4
        let outBytes = CH * SP * 4
        let kernel: ANEKernel
        do {
            kernel = try aneCompile(
                milText: mil,
                weights: [("@model_path/weights/weight.bin", weightBlob)],
                inputBytes:  [inBytes],
                outputBytes: [outBytes]
            )
        } catch {
            slog("[conv] compile FAIL: \(error)"); exit(3)
        }
        slog("[conv] compiled, compile_count=\(aneCompileCount())")

        var input = [Float](repeating: 0, count: CH * SP)
        for i in 0..<(CH * SP) {
            input[i] = Float(i + 1)
        }

        do {
            try input.withUnsafeBytes { buf in
                try kernel.writeInput(0, buf.baseAddress!, bytes: buf.count)
            }
            try kernel.eval()
        } catch {
            slog("[conv] eval FAIL: \(error)"); exit(4)
        }

        var output = [Float](repeating: 0, count: CH * SP)
        do {
            try output.withUnsafeMutableBytes { buf in
                try kernel.readOutput(0, into: buf.baseAddress!, bytes: buf.count)
            }
        } catch {
            slog("[conv] read FAIL: \(error)"); exit(5)
        }

        var maxErr: Float = 0
        var firstFail = -1
        for i in 0..<(CH * SP) {
            let expected = input[i]  // identity
            let err = abs(output[i] - expected)
            if err > maxErr { maxErr = err }
            if err > max(Float(1e-2), Float(0.005) * expected) && firstFail < 0 {
                firstFail = i
            }
        }

        slog("[conv] input[0..4]:  \(Array(input[0..<4]))")
        slog("[conv] output[0..4]: \(Array(output[0..<4]))")
        slog("[conv] expected:     [1.0, 2.0, 3.0, 4.0] (identity)")
        slog("[conv] max abs err:  \(maxErr)")

        if firstFail >= 0 {
            let got = output[firstFail]
            let exp = input[firstFail]
            slog("[conv] FAIL at idx \(firstFail): got \(got), expected \(exp)")
            exit(6)
        }
        slog("[conv] OK")
    }
}
