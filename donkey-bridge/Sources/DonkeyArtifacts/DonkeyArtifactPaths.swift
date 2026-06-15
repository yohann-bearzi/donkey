import Foundation
import CryptoKit

/// Resolves the on-disk layout of donkey's artifacts.
///
/// Donkey lives in a `donkey/` subdirectory of the trunk it was trained
/// against. See docs/DONKEY_ARTIFACT_LAYOUT.md.
public struct DonkeyArtifactPaths: Sendable {
    public let trunkDir: URL
    public let donkeyDir: URL

    public init(trunkDir: URL) {
        self.trunkDir = trunkDir
        self.donkeyDir = trunkDir.appendingPathComponent("donkey", isDirectory: true)
    }

    // MARK: - File paths

    public var manifest:     URL { donkeyDir.appendingPathComponent("donkey.json") }
    public var weights:      URL { donkeyDir.appendingPathComponent("weights.bin") }
    public var weightsSlow:  URL { donkeyDir.appendingPathComponent("weights_slow.bin") }
    public var adamM:        URL { donkeyDir.appendingPathComponent("adam_m.bin") }
    public var adamV:        URL { donkeyDir.appendingPathComponent("adam_v.bin") }
    public var calibration:  URL { donkeyDir.appendingPathComponent("calibration.bin") }
    public var stats:        URL { donkeyDir.appendingPathComponent("stats.json") }
    public var trunkBinding: URL { donkeyDir.appendingPathComponent("trunk_binding.json") }

    // MARK: - Existence checks

    /// True if a donkey has been initialized at this trunk.
    /// (Manifest + slow-copy weights present.)
    public var isInitialized: Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: manifest.path)
            && fm.fileExists(atPath: weightsSlow.path)
    }

    /// True if the donkey directory exists at all.
    public var donkeyDirExists: Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: donkeyDir.path, isDirectory: &isDir)
        return exists && isDir.boolValue
    }

    /// Create the donkey directory if missing. Idempotent.
    public func ensureDonkeyDirExists() throws {
        try FileManager.default.createDirectory(
            at: donkeyDir, withIntermediateDirectories: true)
    }
}

/// Computes a stable hash binding donkey weights to a specific trunk
/// version. SHA256 over (config.json contents + sorted list of
/// safetensors filenames). Cheap, stable across machines, sensitive
/// to trunk changes that matter without reading 96GB of weights.
public enum TrunkHash {
    public static func compute(trunkDir: URL) throws -> String {
        let fm = FileManager.default

        // 1. config.json contents
        let configURL = trunkDir.appendingPathComponent("config.json")
        let configData: Data
        if fm.fileExists(atPath: configURL.path) {
            configData = try Data(contentsOf: configURL)
        } else {
            configData = Data()
        }

        // 2. Sorted list of *.safetensors filenames (names only; reading
        //    contents would defeat the cheap-hash goal)
        let contents = (try? fm.contentsOfDirectory(atPath: trunkDir.path)) ?? []
        let safetensors = contents
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        let listingData = Data(safetensors.joined(separator: "\n").utf8)

        var combined = Data()
        combined.append(configData)
        combined.append(0x0A)  // newline separator
        combined.append(listingData)

        let digest = SHA256.hash(data: combined)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "sha256:" + hex
    }
}
