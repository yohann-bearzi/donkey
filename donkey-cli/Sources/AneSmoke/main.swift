import Foundation
import DonkeyArtifacts
import DonkeyANE

fileprivate func slog(_ s: String) {
    FileHandle.standardError.write(Data("\(s)\n".utf8))
}

@main
struct AneSmoke {
    static func main() {
        slog("[smoke] AneSmoke -- MLX-free")
        let trunkDir = URL(fileURLWithPath:
            ProcessInfo.processInfo.environment["MIMO_PATH"]
            ?? "/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M")
        let paths = DonkeyArtifactPaths(trunkDir: trunkDir)
        slog("[paths] trunk:       \(paths.trunkDir.path)")
        slog("[paths] donkey:      \(paths.donkeyDir.path)")
        slog("[paths] dir exists:  \(paths.donkeyDirExists)")
        slog("[paths] initialized: \(paths.isInitialized)")
        do {
            let hash = try TrunkHash.compute(trunkDir: trunkDir)
            slog("[hash]  trunk hash: \(hash.prefix(24))...")
        } catch {
            slog("[hash]  FAIL: \(error)")
            exit(10)
        }
        slog("[ane]   calling aneBridgeInit()...")
        do {
            try aneBridgeInit()
            slog("[ane]   bridge init:   ok")
            slog("[ane]   compile count: \(aneCompileCount())")
        } catch {
            slog("[ane]   FAIL: \(error)")
            exit(11)
        }
        slog("[smoke] OK")
    }
}
