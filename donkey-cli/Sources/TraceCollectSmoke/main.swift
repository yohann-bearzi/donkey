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

/// Phase 5: dump (lastHiddenState[t], token[t]) per position to streaming .bin files.
///
/// Single forward, teacher-forced on a synthetic 64-token sequence. Validates the
/// trace-collection pipeline end-to-end without any tokenizer / corpus dependency.
/// Real prompts get layered on once round-trip is proven.
///
/// Files written (append-mode for hidden + tokens, overwrite for meta):
///   - lastHiddenState.bin   fp32, T * 4096 floats per call
///   - tokens.bin            int32, T ints per call
///   - meta.json             {hidden_dim, dtypes, total_positions, trunk_hash}
///
/// Usage: trace-collect-smoke [output_dir]    (default /tmp/donkey_trace)
@main
struct TraceCollectSmoke {
    static func main() async {
        let args = CommandLine.arguments
        let outDirStr = args.count > 1 ? args[1] : "/tmp/donkey_trace"
        let outDir = URL(fileURLWithPath: outDirStr)

        let modelPath = ProcessInfo.processInfo.environment["MIMO_PATH"]
            ?? "/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M"
        let trunkDir = URL(fileURLWithPath: modelPath)

        do {
            try FileManager.default.createDirectory(
                at: outDir, withIntermediateDirectories: true)
        } catch {
            slog("[trace] FAIL: mkdir \(outDirStr): \(error)"); exit(1)
        }
        slog("[trace] outDir: \(outDir.path)")

        let trunkHash: String
        do {
            trunkHash = try TrunkHash.compute(trunkDir: trunkDir)
            slog("[trace] trunk hash: \(trunkHash.prefix(24))...")
        } catch {
            slog("[trace] FAIL: trunk hash: \(error)"); exit(2)
        }

        // Synthetic 64-token sequence; deterministic, safely inside vocab.
        let T = 64
        let synthTokens: [Int32] = (0..<T).map { Int32(($0 * 17 + 13) % 32000) }

        slog("[trace] loading MiMo from \(modelPath)")
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: trunkDir,
                using: NoOpTokenizerLoader()
            )

            await container.perform { ctx in
                let toks = MLXArray(synthTokens).reshaped([1, T])
                let input = LMInput.Text(tokens: toks)

                guard let model = ctx.model as? MiMoV2FlashModel else {
                    slog("[trace] FAIL: not MiMoV2FlashModel"); exit(3)
                }

                let out = model(input, cache: nil, state: nil)
                slog("[trace] logits shape: \(out.logits.shape)")

                guard let st = out.state, let lh = st.lastHiddenState else {
                    slog("[trace] FAIL: lastHiddenState nil"); exit(4)
                }
                slog("[trace] lastHiddenState shape: \(lh.shape)")

                // [1, T, 4096] (fp16/bf16) -> flat [T*4096] fp32
                let hiddenFloats: [Float] = lh.asType(.float32).asArray(Float.self)
                let expected = T * 4096
                guard hiddenFloats.count == expected else {
                    slog("[trace] FAIL: hidden count \(hiddenFloats.count) != \(expected)")
                    exit(5)
                }

                let hiddenURL = outDir.appendingPathComponent("lastHiddenState.bin")
                let tokensURL = outDir.appendingPathComponent("tokens.bin")
                let metaURL   = outDir.appendingPathComponent("meta.json")

                let hiddenData = hiddenFloats.withUnsafeBufferPointer { Data(buffer: $0) }
                let tokensData = synthTokens.withUnsafeBufferPointer { Data(buffer: $0) }

                do {
                    try appendData(hiddenData, to: hiddenURL)
                    try appendData(tokensData, to: tokensURL)
                } catch {
                    slog("[trace] FAIL: append: \(error)"); exit(6)
                }

                // File-size cross-check: sanity that the two streams agree on T_total
                let hsz: Int
                let tsz: Int
                do {
                    let ha = try FileManager.default.attributesOfItem(atPath: hiddenURL.path)
                    let ta = try FileManager.default.attributesOfItem(atPath: tokensURL.path)
                    hsz = (ha[.size] as? Int) ?? 0
                    tsz = (ta[.size] as? Int) ?? 0
                } catch {
                    slog("[trace] FAIL: stat: \(error)"); exit(7)
                }

                let totalPositions = tsz / MemoryLayout<Int32>.size
                let expectedHiddenBytes = totalPositions * 4096 * MemoryLayout<Float>.size
                guard hsz == expectedHiddenBytes else {
                    slog("[trace] FAIL: size mismatch hidden=\(hsz) expected=\(expectedHiddenBytes)")
                    exit(8)
                }

                // meta.json (overwrite each run with current totals)
                let meta: [String: Any] = [
                    "hidden_dim": 4096,
                    "hidden_dtype": "float32",
                    "token_dtype": "int32",
                    "total_positions": totalPositions,
                    "trunk_hash": trunkHash,
                ]
                do {
                    let metaData = try JSONSerialization.data(
                        withJSONObject: meta,
                        options: [.prettyPrinted, .sortedKeys])
                    try metaData.write(to: metaURL)
                } catch {
                    slog("[trace] FAIL: meta write: \(error)"); exit(9)
                }

                slog("[trace] appended T=\(T); total positions on disk: \(totalPositions)")
                slog("[trace] OK")
            }
        } catch {
            slog("[trace] error: \(error)"); exit(99)
        }
    }
}
