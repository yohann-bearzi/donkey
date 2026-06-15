// DonkeyConfig — canonical configuration for the donkey world-model drafter.
//
// Loaded from a JSON spec (single source of truth shared with PyTorch).
// All architectural sizes derive from this struct; the orchestrator and the
// kernel compilation pipeline both build from it. No magic numbers anywhere
// downstream.
//
// To run a parameter sweep: copy spec/donkey_v2_default.json, edit knobs,
// instantiate orchestrator with the new config. No code changes needed.
import Foundation

public enum DonkeyConfigError: Error, CustomStringConvertible {
    case invalidJSON(String)
    case invariantViolated(String)
    case unsupportedValue(String)

    public var description: String {
        switch self {
        case .invalidJSON(let s):       return "DonkeyConfig: invalid JSON: \(s)"
        case .invariantViolated(let s): return "DonkeyConfig: invariant violated: \(s)"
        case .unsupportedValue(let s):  return "DonkeyConfig: unsupported value: \(s)"
        }
    }
}

public struct DonkeyConfig: Codable, Equatable {
    public var schemaVersion: Int
    public var modelName: String
    public var trunk: TrunkConfig
    public var architecture: ArchitectureConfig
    public var sequence: SequenceConfig
    public var training: TrainingConfig

    public struct TrunkConfig: Codable, Equatable {
        public var hiddenDim: Int       // 4096 for MiMo-V2-Flash
        public var vocabSize: Int       // 152576
        public var name: String         // "MiMo-V2-Flash"

        enum CodingKeys: String, CodingKey {
            case hiddenDim = "hidden_dim"
            case vocabSize = "vocab_size"
            case name
        }
    }

    public struct ArchitectureConfig: Codable, Equatable {
        public var hiddenDim: Int       // donkey's working dim, e.g. 1024
        public var ffnDim: Int          // FFN inner dim, e.g. 4096
        public var heads: Int           // 16
        public var headDim: Int         // hiddenDim / heads, e.g. 64
        public var nLayers: Int         // 2
        public var outConfDim: Int      // 1 confidence channel
        public var ffnActivation: String  // "silu" or "swiglu"
        public var rmsEps: Float        // 1e-6
        // Rollout depth: 1 = single forward (parallel-K). >1 = tree decoding,
        // re-feeding donkey's own predicted hiddens for deeper drafts.
        // Currently only depth=1 implemented; field is parsed and validated
        // so configs declaring intent are accepted without code changes.
        public var rolloutDepth: Int

        enum CodingKeys: String, CodingKey {
            case hiddenDim    = "hidden_dim"
            case ffnDim       = "ffn_dim"
            case heads
            case headDim      = "head_dim"
            case nLayers      = "n_layers"
            case outConfDim   = "out_conf_dim"
            case ffnActivation = "ffn_activation"
            case rmsEps       = "rms_eps"
            case rolloutDepth = "rollout_depth"
        }
    }

    public struct SequenceConfig: Codable, Equatable {
        public var windowSize: Int      // W: history slots
        public var draftSize: Int       // K: prediction slots
        public var spatialPad: Int      // SP = next ANE-safe size >= W+K

        enum CodingKeys: String, CodingKey {
            case windowSize = "window_size"
            case draftSize  = "draft_size"
            case spatialPad = "spatial_pad"
        }
    }

    public struct TrainingConfig: Codable, Equatable {
        public var lossL2: Float
        public var lossCos: Float
        public var lossCe: Float
        public var lossCalib: Float
        public var learningRate: Float
        public var lrWarmupSteps: Int
        public var lrFinal: Float
        public var ewmaDecay: Float

        enum CodingKeys: String, CodingKey {
            case lossL2        = "loss_l2"
            case lossCos       = "loss_cos"
            case lossCe        = "loss_ce"
            case lossCalib     = "loss_calib"
            case learningRate  = "learning_rate"
            case lrWarmupSteps = "lr_warmup_steps"
            case lrFinal       = "lr_final"
            case ewmaDecay     = "ewma_decay"
        }
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelName     = "model_name"
        case trunk
        case architecture
        case sequence
        case training
    }

    // === Loaders ===

    public init(jsonURL: URL) throws {
        let data = try Data(contentsOf: jsonURL)
        try self.init(jsonData: data)
    }

    public init(jsonData: Data) throws {
        let decoder = JSONDecoder()
        do {
            self = try decoder.decode(DonkeyConfig.self, from: jsonData)
        } catch {
            throw DonkeyConfigError.invalidJSON(String(describing: error))
        }
        try validate()
    }

    // === Invariants ===
    //
    // Enforced at load time. Fail fast: the orchestrator never sees a config
    // that violates any of these.
    public func validate() throws {
        if schemaVersion != 1 {
            throw DonkeyConfigError.unsupportedValue(
                "schema_version: expected 1, got \(schemaVersion)")
        }

        // hidden_dim must equal heads * head_dim (multi-head split).
        if architecture.hiddenDim != architecture.heads * architecture.headDim {
            throw DonkeyConfigError.invariantViolated(
                "architecture.hidden_dim (\(architecture.hiddenDim)) != heads (\(architecture.heads)) * head_dim (\(architecture.headDim))")
        }
        if architecture.heads <= 0 || architecture.headDim <= 0 || architecture.nLayers <= 0 {
            throw DonkeyConfigError.invariantViolated("architecture: heads/head_dim/n_layers must be > 0")
        }
        if architecture.rolloutDepth < 1 {
            throw DonkeyConfigError.invariantViolated(
                "architecture.rollout_depth: must be >= 1, got \(architecture.rolloutDepth)")
        }

        // Window + draft must fit in the spatial pad.
        let wPlusK = sequence.windowSize + sequence.draftSize
        if wPlusK > sequence.spatialPad {
            throw DonkeyConfigError.invariantViolated(
                "sequence: window_size (\(sequence.windowSize)) + draft_size (\(sequence.draftSize)) > spatial_pad (\(sequence.spatialPad))")
        }

        // ANE-validated SP values only. (Measured: SP=8 fails SDPA at eval; SP=16,32,64,128 work.)
        let allowedSP = [16, 32, 64, 128]
        if !allowedSP.contains(sequence.spatialPad) {
            throw DonkeyConfigError.unsupportedValue(
                "sequence.spatial_pad: must be one of \(allowedSP), got \(sequence.spatialPad)")
        }
        if sequence.windowSize <= 0 || sequence.draftSize <= 0 {
            throw DonkeyConfigError.invariantViolated("sequence: window_size and draft_size must be > 0")
        }

        // FFN activation: only silu is wired in v2.0; swiglu deferred.
        if architecture.ffnActivation != "silu" {
            throw DonkeyConfigError.unsupportedValue(
                "architecture.ffn_activation: only 'silu' supported in v2.0, got '\(architecture.ffnActivation)'")
        }

        // Trunk binding sanity.
        if trunk.hiddenDim <= 0 || trunk.vocabSize <= 0 {
            throw DonkeyConfigError.invariantViolated("trunk: dimensions must be > 0")
        }

        // Training hyperparameters — non-negative.
        let weights = [training.lossL2, training.lossCos, training.lossCe, training.lossCalib]
        if weights.contains(where: { $0 < 0 }) {
            throw DonkeyConfigError.invariantViolated("training: all loss weights must be >= 0")
        }
        if training.learningRate <= 0 || training.lrFinal <= 0 || training.lrWarmupSteps < 0 {
            throw DonkeyConfigError.invariantViolated("training: learning rates must be > 0, warmup >= 0")
        }
        if training.ewmaDecay < 0 || training.ewmaDecay > 1 {
            throw DonkeyConfigError.invariantViolated(
                "training.ewma_decay: must be in [0,1], got \(training.ewmaDecay)")
        }
    }

    // === Convenience derived sizes ===
    public var qkvCh: Int { 3 * architecture.hiddenDim }
    // Head output: trunk-sized predicted hidden + confidence channel(s).
    // The head projects donkey's working dim back UP to trunk.hiddenDim
    // so it can be fed through the trunk's lm_head for verification.
    public var outCh: Int { trunk.hiddenDim + architecture.outConfDim }
}
