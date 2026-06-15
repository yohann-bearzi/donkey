import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import BenchmarkHelpers
import DonkeyArtifacts

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

fileprivate func appendData(_ data: Data, to url: URL) throws {
    if FileManager.default.fileExists(atPath: url.path) {
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }
        try fh.seekToEnd()
        try fh.write(contentsOf: data)
    } else {
        try data.write(to: url)
    }
}

/// Phase 5.2 (KV-cached): trunk-self-generated trace.
///
/// Prefill: one forward over the full prompt against a fresh KV cache.
/// Decode: each step feeds a single new token; the cache holds all prior
/// K/V. Per-step compute drops from O(T) to O(1) on the trunk side, which
/// makes corpus-scale collection (thousands of tokens) realistic.
///
/// Capture semantics identical to the cacheless smoke: each position pairs
/// the input token fed at that position with the residual the trunk
/// produced after consuming it. Greedy decode (argmax).
///
/// Usage: trace-selfgen-smoke [output_dir] [maxNew=16] [seed=42]
@main
struct TraceSelfGenSmoke {
    static func main() async {
        let args = CommandLine.arguments
        let outDirStr = args.count > 1 ? args[1] : "/tmp/donkey_selfgen"
        let maxNew    = args.count > 2 ? (Int(args[2]) ?? 16) : 16
        let seed      = args.count > 3 ? (Int(args[3]) ?? 42) : 42

        let outDir = URL(fileURLWithPath: outDirStr)
        let modelPath = ProcessInfo.processInfo.environment["MIMO_PATH"]
            ?? "/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M"
        let trunkDir = URL(fileURLWithPath: modelPath)

        do {
            try FileManager.default.createDirectory(
                at: outDir, withIntermediateDirectories: true)
        } catch {
            slog("[selfgen] FAIL: mkdir: \(error)"); exit(1)
        }
        slog("[selfgen] outDir: \(outDirStr)  maxNew: \(maxNew)  seed: \(seed)")

        let trunkHash: String
        do {
            trunkHash = try TrunkHash.compute(trunkDir: trunkDir)
        } catch {
            slog("[selfgen] FAIL: hash: \(error)"); exit(2)
        }
        slog("[selfgen] trunk hash: \(trunkHash.prefix(24))...")

        let promptTokens: [Int32] = (0..<4).map { i in
            Int32(((seed + i * 1337) * 17 + 13) % 32000)
        }
        slog("[selfgen] prompt: \(promptTokens)")

        slog("[selfgen] loading MiMo from \(modelPath)")
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: trunkDir,
                using: NoOpTokenizerLoader()
            )

            await container.perform { ctx in
                guard let model = ctx.model as? MiMoV2FlashModel else {
                    slog("[selfgen] FAIL: not MiMoV2FlashModel"); exit(3)
                }

                let params = GenerateParameters()
                let cache = model.newCache(parameters: params)
                slog("[selfgen] cache initialised: \(cache.count) layer(s)")

                var hiddenFlat: [Float] = []
                var tokensCaptured: [Int32] = []
                hiddenFlat.reserveCapacity((promptTokens.count + maxNew) * 4096)
                tokensCaptured.reserveCapacity(promptTokens.count + maxNew)

                let t0 = Date()

                // === PREFILL: full prompt into cache ===
                var nextTok: Int
                do {
                    let arr = MLXArray(promptTokens).reshaped([1, promptTokens.count])
                    let result = model(
                        LMInput.Text(tokens: arr), cache: cache, state: nil)

                    guard let st = result.state, let lh = st.lastHiddenState else {
                        slog("[selfgen] FAIL: prefill hidden nil"); exit(4)
                    }
                    let lhFlat: [Float] = lh.asType(.float32).asArray(Float.self)
                    let expected = promptTokens.count * 4096
                    guard lhFlat.count == expected else {
                        slog("[selfgen] FAIL: prefill count \(lhFlat.count) != \(expected)")
                        exit(5)
                    }
                    hiddenFlat.append(contentsOf: lhFlat)
                    tokensCaptured.append(contentsOf: promptTokens)

                    nextTok = argMax(result.logits[0, -1, 0...], axis: -1).item(Int.self)
                    slog("[selfgen] prefill T=\(promptTokens.count); first sampled=\(nextTok)")
                }
                let tPrefill = Date().timeIntervalSince(t0)

                // === DECODE: single-token forwards reusing cache ===
                let tDecodeStart = Date()
                for step in 0..<maxNew {
                    let arr = MLXArray([Int32(nextTok)]).reshaped([1, 1])
                    let result = model(
                        LMInput.Text(tokens: arr), cache: cache, state: nil)

                    guard let st = result.state, let lh = st.lastHiddenState else {
                        slog("[selfgen] FAIL: step \(step) hidden nil"); exit(6)
                    }
                    let lhFlat: [Float] = lh.asType(.float32).asArray(Float.self)
                    guard lhFlat.count == 4096 else {
                        slog("[selfgen] FAIL: step \(step) hidden size \(lhFlat.count)")
                        exit(7)
                    }
                    hiddenFlat.append(contentsOf: lhFlat)
                    tokensCaptured.append(Int32(nextTok))

                    if step < maxNew - 1 {
                        nextTok = argMax(result.logits[0, -1, 0...], axis: -1).item(Int.self)
                    }

                    if step < 3 || step % 16 == 0 || step == maxNew - 1 {
                        slog("[selfgen] step \(step): fed=\(tokensCaptured.last!)")
                    }
                }
                let tDecode = Date().timeIntervalSince(tDecodeStart)

                slog(String(format: "[selfgen] timings: prefill=%.2fs decode=%.2fs tok/s=%.1f",
                    tPrefill, tDecode, Double(maxNew) / tDecode))

                let hiddenURL = outDir.appendingPathComponent("lastHiddenState.bin")
                let tokensURL = outDir.appendingPathComponent("tokens.bin")
                let metaURL   = outDir.appendingPathComponent("meta.json")

                let hiddenData = hiddenFlat.withUnsafeBufferPointer { Data(buffer: $0) }
                let tokensData = tokensCaptured.withUnsafeBufferPointer { Data(buffer: $0) }
                do {
                    try appendData(hiddenData, to: hiddenURL)
                    try appendData(tokensData, to: tokensURL)
                } catch {
                    slog("[selfgen] FAIL: append: \(error)"); exit(8)
                }

                let hsz: Int
                let tsz: Int
                do {
                    let ha = try FileManager.default.attributesOfItem(atPath: hiddenURL.path)
                    let ta = try FileManager.default.attributesOfItem(atPath: tokensURL.path)
                    hsz = (ha[.size] as? Int) ?? 0
                    tsz = (ta[.size] as? Int) ?? 0
                } catch {
                    slog("[selfgen] FAIL: stat: \(error)"); exit(9)
                }
                let totalPositions = tsz / MemoryLayout<Int32>.size
                let expectedHiddenBytes = totalPositions * 4096 * MemoryLayout<Float>.size
                guard hsz == expectedHiddenBytes else {
                    slog("[selfgen] FAIL: size mismatch hidden=\(hsz) expected=\(expectedHiddenBytes)")
                    exit(10)
                }

                let meta: [String: Any] = [
                    "hidden_dim": 4096,
                    "hidden_dtype": "float32",
                    "token_dtype": "int32",
                    "total_positions": totalPositions,
                    "trunk_hash": trunkHash,
                    "source": "self-generation (greedy, KV-cached)",
                ]
                do {
                    let metaData = try JSONSerialization.data(
                        withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
                    try metaData.write(to: metaURL)
                } catch {
                    slog("[selfgen] FAIL: meta: \(error)"); exit(11)
                }

                slog("[selfgen] OK")
            }
        } catch {
            slog("[selfgen] error: \(error)"); exit(99)
        }
    }
}
