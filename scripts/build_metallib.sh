#!/bin/sh
# Build the mlx-swift Metal library bundle that the donkey binary needs
# at runtime. SwiftPM builds don't produce this; only xcodebuild does.
# Run once per machine, or after blowing away DerivedData.
set -e

cd ~/third-party/vmlx-swift-lm
mkdir -p /tmp/mlx-derived
xcodebuild build \
    -scheme MLXLLM \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath /tmp/mlx-derived \
    -quiet

BUNDLE=$(find /tmp/mlx-derived -name "mlx-swift_Cmlx.bundle" -type d | head -1)
if [ -z "$BUNDLE" ]; then
    echo "ERROR: bundle not produced"
    exit 1
fi

cp -R "$BUNDLE" ~/projects/donkey/.build/arm64-apple-macosx/release/
echo "OK: bundle at ~/projects/donkey/.build/arm64-apple-macosx/release/mlx-swift_Cmlx.bundle"
