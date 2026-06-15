import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import BenchmarkHelpers
import DonkeyArtifacts
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

@main
struct DonkeySmoke {
    static func main() async {
        let modelPath = ProcessInfo.processInfo.environment["MIMO_PATH"]
            ?? "/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M"
        let trunkDir = URL(fileURLWithPath: modelPath)

        // === Artifact layout smoke ===
        let paths = DonkeyArtifactPaths(trunkDir: trunkDir)
        slog("[paths] donkey:        \(paths.donkeyDir.path)")
        slog("[paths] dir exists:    \(paths.donkeyDirExists)")
        slog("[paths] initialized:   \(paths.isInitialized)")

        // === Trunk hash ===
        do {
            let hash = try TrunkHash.compute(trunkDir: trunkDir)
            slog("[hash]  trunk hash:    \(hash.prefix(24))...")
        } catch {
            slog("[hash]  FAIL: \(error)")
            exit(10)
        }

        // === ANE bridge init (NEW) ===
        do {
            try aneBridgeInit()
            slog("[ane]   bridge init:   ok")
            slog("[ane]   compile count: \(aneCompileCount())")
        } catch {
            slog("[ane]   FAIL: \(error)")
            exit(11)
        }

        // === MiMo load + EAGLE-3 taps ===
        slog("[smoke] loading MiMo from \(modelPath)")
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: trunkDir,
                using: NoOpTokenizerLoader()
            )

            await container.perform { ctx in
                let toks = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)])
                    .reshaped([1, 4])
                let input = LMInput.Text(tokens: toks)

                guard let model = ctx.model as? MiMoV2FlashModel else {
                    slog("[smoke] FAIL: not MiMoV2FlashModel"); exit(1)
                }

                let out = model(input, cache: nil, state: nil)
                slog("[smoke] logits shape: \(out.logits.shape)")

                guard let st = out.state else { slog("[smoke] FAIL: state nil"); exit(2) }
                if let lh = st.lastHiddenState {
                    slog("[smoke] lastHiddenState shape: \(lh.shape)")
                } else {
                    slog("[smoke] FAIL: lastHiddenState nil"); exit(3)
                }
                guard let taps = st.intermediateHiddenStates else {
                    slog("[smoke] FAIL: taps nil"); exit(4)
                }
                slog("[smoke] taps count: \(taps.count)")
                guard taps.count == 3 else {
                    slog("[smoke] FAIL: expected 3, got \(taps.count)"); exit(5)
                }
                slog("[smoke] OK")
            }
        } catch {
            slog("[smoke] error: \(error)")
            exit(99)
        }
    }
}
