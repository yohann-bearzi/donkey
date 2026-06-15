// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "donkey",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "donkey",    targets: ["DonkeyCLI"]),
        .executable(name: "ane-smoke",      targets: ["AneSmoke"]),
        .executable(name: "ane-conv-smoke", targets: ["AneConvSmoke"]),
        .executable(name: "ane-tap-fuse-smoke", targets: ["AneTapFuseSmoke"]),
        .executable(name: "ane-rect-conv-smoke", targets: ["AneRectConvSmoke"]),
        .executable(name: "ane-tri-weight-smoke", targets: ["AneTriWeightSmoke"]),
        .executable(name: "ane-linear1x1-smoke", targets: ["AneLinear1x1Smoke"]),
        .executable(name: "ane-rmsnorm-smoke", targets: ["AneRMSNormSmoke"]),
        .executable(name: "ane-roundtrip-bench", targets: ["AneRoundTripBench"]),
        .executable(name: "ane-sdpa-smoke", targets: ["AneSDPASmoke"]),
        .executable(name: "ane-silu-smoke", targets: ["AneSiLUSmoke"]),
        .executable(name: "cpu-silu-smoke", targets: ["CpuSiLUSmoke"]),
        .executable(name: "cpu-rmsnorm-bench", targets: ["CpuRMSNormBench"]),
        .executable(name: "cpu-silu-cheby-bench", targets: ["CpuSiLUChebyBench"]),
        .executable(name: "ops-smoke", targets: ["OpsSmoke"]),
        .executable(name: "config-smoke", targets: ["ConfigSmoke"]),
        .executable(name: "ops-tests", targets: ["OpsTests"]),
        .executable(name: "world-compile-smoke", targets: ["WorldForwardCompileSmoke"]),
        .executable(name: "world-forward-smoke", targets: ["WorldForwardSmoke"]),
        .executable(name: "world-dump-smoke", targets: ["WorldDumpSmoke"]),
        .executable(name: "trace-collect-smoke", targets: ["TraceCollectSmoke"]),
        .executable(name: "trace-selfgen-smoke", targets: ["TraceSelfGenSmoke"]),
        .executable(name: "trace-selfgen-batch", targets: ["TraceSelfGenBatch"]),
        .library(name: "DonkeyArtifacts", targets: ["DonkeyArtifacts"]),
        .library(name: "DonkeyANE",       targets: ["DonkeyANE"]),
        .library(name: "DonkeyOps",       targets: ["DonkeyOps"]),
        .library(name: "DonkeyRuntime",   targets: ["DonkeyRuntime"]),
        .library(name: "DonkeyConfig",    targets: ["DonkeyConfig"]),
    ],
    dependencies: [
        .package(path: "../../third-party/vmlx-swift-lm"),
    ],
    targets: [
        .systemLibrary(
            name: "CANEBridge",
            path: "donkey-trainer/ane-module"
        ),
        .target(
            name: "DonkeyANE",
            dependencies: ["CANEBridge"],
            path: "donkey-bridge/Sources/DonkeyANE",
            linkerSettings: [
                .unsafeFlags([
                    "-Ldonkey-trainer/ane",
                    "-Xlinker", "-rpath",
                    "-Xlinker", "@loader_path/../donkey-trainer/ane",
                ]),
                .linkedLibrary("ane_bridge"),
            ]
        ),
        .target(
            name: "DonkeyArtifacts",
            path: "donkey-bridge/Sources/DonkeyArtifacts"
        ),
        .target(
            name: "DonkeyOps",
            path: "donkey-runtime/Sources/DonkeyOps"
        ),
        .target(
            name: "DonkeyConfig",
            path: "donkey-runtime/Sources/DonkeyConfig"
        ),
        .target(
            name: "DonkeyRuntime",
            dependencies: ["DonkeyANE", "DonkeyOps", "DonkeyConfig"],
            path: "donkey-runtime/Sources/DonkeyRuntime"
        ),

        .executableTarget(
            name: "DonkeyCLI",
            dependencies: [
                .product(name: "MLXLMCommon",      package: "vmlx-swift-lm"),
                .product(name: "MLXLLM",           package: "vmlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "vmlx-swift-lm"),
                "DonkeyArtifacts",
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/DonkeyCLI",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneSmoke",
            dependencies: [
                "DonkeyArtifacts",
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),

        .executableTarget(
            name: "AneConvSmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneConvSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneTapFuseSmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneTapFuseSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneRectConvSmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneRectConvSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneTriWeightSmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneTriWeightSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneLinear1x1Smoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneLinear1x1Smoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneRMSNormSmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneRMSNormSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneRoundTripBench",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneRoundTripBench",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneSDPASmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneSDPASmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "AneSiLUSmoke",
            dependencies: [
                "DonkeyANE",
            ],
            path: "donkey-cli/Sources/AneSiLUSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "CpuSiLUSmoke",
            dependencies: [],
            path: "donkey-cli/Sources/CpuSiLUSmoke"
        ),
        .executableTarget(
            name: "CpuRMSNormBench",
            dependencies: [],
            path: "donkey-cli/Sources/CpuRMSNormBench"
        ),
        .executableTarget(
            name: "CpuSiLUChebyBench",
            dependencies: [],
            path: "donkey-cli/Sources/CpuSiLUChebyBench"
        ),
        .executableTarget(
            name: "OpsSmoke",
            dependencies: ["DonkeyOps"],
            path: "donkey-cli/Sources/OpsSmoke"
        ),
        .executableTarget(
            name: "ConfigSmoke",
            dependencies: ["DonkeyConfig"],
            path: "donkey-cli/Sources/ConfigSmoke"
        ),
        .executableTarget(
            name: "OpsTests",
            dependencies: ["DonkeyOps"],
            path: "donkey-cli/Sources/OpsTests"
        ),
        .executableTarget(
            name: "WorldForwardCompileSmoke",
            dependencies: ["DonkeyConfig", "DonkeyRuntime"],
            path: "donkey-cli/Sources/WorldForwardCompileSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "WorldForwardSmoke",
            dependencies: ["DonkeyConfig", "DonkeyRuntime"],
            path: "donkey-cli/Sources/WorldForwardSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "WorldDumpSmoke",
            dependencies: ["DonkeyConfig", "DonkeyRuntime"],
            path: "donkey-cli/Sources/WorldDumpSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "TraceCollectSmoke",
            dependencies: [
                .product(name: "MLXLMCommon",      package: "vmlx-swift-lm"),
                .product(name: "MLXLLM",           package: "vmlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "vmlx-swift-lm"),
                "DonkeyArtifacts",
            ],
            path: "donkey-cli/Sources/TraceCollectSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "TraceSelfGenSmoke",
            dependencies: [
                .product(name: "MLXLMCommon",      package: "vmlx-swift-lm"),
                .product(name: "MLXLLM",           package: "vmlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "vmlx-swift-lm"),
                "DonkeyArtifacts",
            ],
            path: "donkey-cli/Sources/TraceSelfGenSmoke",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),
        .executableTarget(
            name: "TraceSelfGenBatch",
            dependencies: [
                .product(name: "MLXLMCommon",      package: "vmlx-swift-lm"),
                .product(name: "MLXLLM",           package: "vmlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "vmlx-swift-lm"),
                "DonkeyArtifacts",
            ],
            path: "donkey-cli/Sources/TraceSelfGenBatch",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path",
                    "-Xlinker", "-rpath", "-Xlinker", "@executable_path/../../../donkey-trainer/ane",
                ]),
            ]
        ),

    ]
)