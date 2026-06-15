import Foundation
import CANEBridge

/// Errors raised by the ANE bridge wrapper.
public enum ANEBridgeError: Error, CustomStringConvertible {
    case initFailed
    case compileFailed
    case evalFailed
    case sizeMismatch(expected: Int, got: Int)

    public var description: String {
        switch self {
        case .initFailed:                return "ANE bridge init failed"
        case .compileFailed:             return "ANE kernel compile failed"
        case .evalFailed:                return "ANE kernel eval failed"
        case .sizeMismatch(let e, let g): return "size mismatch: expected \(e) bytes, got \(g)"
        }
    }
}

/// One-time initialization of the ANE bridge.
///
/// Loads the AppleNeuralEngine private framework and resolves the four
/// classes the bridge needs. Idempotent. Safe to call multiple times.
/// Throws ANEBridgeError.initFailed if any private class is missing.
public func aneBridgeInit() throws {
    let rc = ane_bridge_init()
    if rc != 0 { throw ANEBridgeError.initFailed }
}

/// Returns the number of kernels compiled since process start
/// (or last reset). Caller uses this to budget against the ~119-compile
/// per-process limit on ANE.
public func aneCompileCount() -> Int {
    Int(ane_bridge_get_compile_count())
}

public func aneResetCompileCount() {
    ane_bridge_reset_compile_count()
}

/// Owned handle to a compiled ANE kernel. Releases its IOSurface refs,
/// temp dir, and ANE model on deinit.
public final class ANEKernel: @unchecked Sendable {
    @usableFromInline let handle: OpaquePointer
    public let inputBytes:  [Int]
    public let outputBytes: [Int]

    init(
        handle: OpaquePointer,
        inputBytes: [Int],
        outputBytes: [Int]
    ) {
        self.handle = handle
        self.inputBytes = inputBytes
        self.outputBytes = outputBytes
    }

    deinit {
        ane_bridge_free(handle)
    }

    /// Dispatch the kernel synchronously on ANE. Inputs must already be
    /// written via `writeInput`; outputs are readable via `readOutput`
    /// after this returns.
    public func eval() throws {
        if !ane_bridge_eval_logged(handle) {
            throw ANEBridgeError.evalFailed
        }
    }

    /// Copy bytes into the i-th input IOSurface. The buffer count must
    /// equal the size declared at compile time.
    public func writeInput(_ idx: Int, _ data: UnsafeRawPointer, bytes: Int) throws {
        guard idx >= 0, idx < inputBytes.count else {
            throw ANEBridgeError.sizeMismatch(expected: inputBytes[0], got: bytes)
        }
        guard bytes == inputBytes[idx] else {
            throw ANEBridgeError.sizeMismatch(expected: inputBytes[idx], got: bytes)
        }
        ane_bridge_write_input(handle, Int32(idx), data, bytes)
    }

    /// Copy bytes out of the i-th output IOSurface.
    public func readOutput(_ idx: Int, into buffer: UnsafeMutableRawPointer, bytes: Int) throws {
        guard idx >= 0, idx < outputBytes.count else {
            throw ANEBridgeError.sizeMismatch(expected: outputBytes[0], got: bytes)
        }
        guard bytes == outputBytes[idx] else {
            throw ANEBridgeError.sizeMismatch(expected: outputBytes[idx], got: bytes)
        }
        ane_bridge_read_output(handle, Int32(idx), buffer, bytes)
    }
}

/// Compile a MIL program into an ANE kernel.
///
/// - Parameters:
///   - milText: UTF-8 MIL program source.
///   - weights: array of (name, data) tuples; names like
///     `"@model_path/weights/wq.bin"` are referenced from the MIL.
///   - inputBytes: byte size of each input tensor (matches MIL declarations).
///   - outputBytes: byte size of each output tensor.
public func aneCompile(
    milText: String,
    weights: [(name: String, data: Data)],
    inputBytes: [Int],
    outputBytes: [Int]
) throws -> ANEKernel {
    let milData = Array(milText.utf8)
    var weightNames:  [UnsafePointer<CChar>?] = []
    var weightCStrs:  [[CChar]] = []   // own the buffers
    var weightPtrs:   [UnsafePointer<UInt8>?] = []
    var weightSizes:  [Int] = []
    var weightDatas:  [Data] = []      // own the bytes

    for (name, data) in weights {
        let cstr = name.utf8CString.map { CChar($0) }
        weightCStrs.append(cstr)
        weightNames.append(weightCStrs[weightCStrs.count - 1].withUnsafeBufferPointer { $0.baseAddress })
        weightDatas.append(data)
        weightSizes.append(data.count)
    }
    // Build the data-pointer array after Data instances are stable in the array
    for d in weightDatas {
        weightPtrs.append(d.withUnsafeBytes { $0.bindMemory(to: UInt8.self).baseAddress })
    }

    var inSizes  = inputBytes
    var outSizes = outputBytes

    let handle: OpaquePointer? = milData.withUnsafeBufferPointer { milBuf in
        weightNames.withUnsafeMutableBufferPointer { namesBuf in
            weightPtrs.withUnsafeMutableBufferPointer { ptrsBuf in
                weightSizes.withUnsafeMutableBufferPointer { sizesBuf in
                    inSizes.withUnsafeMutableBufferPointer { inBuf in
                        outSizes.withUnsafeMutableBufferPointer { outBuf in
                            ane_bridge_compile_multi_weights(
                                milBuf.baseAddress, milData.count,
                                namesBuf.baseAddress, ptrsBuf.baseAddress,
                                sizesBuf.baseAddress, Int32(weights.count),
                                Int32(inputBytes.count), inBuf.baseAddress,
                                Int32(outputBytes.count), outBuf.baseAddress
                            )
                        }
                    }
                }
            }
        }
    }

    guard let h = handle else { throw ANEBridgeError.compileFailed }
    return ANEKernel(handle: h, inputBytes: inputBytes, outputBytes: outputBytes)
}

/// Build an fp16 weight blob in ANE format from float32 input.
/// The returned Data owns the bytes; safe to pass to `aneCompile`.
public func aneBuildWeightBlobFP16(_ src: [Float], rows: Int, cols: Int) -> Data {
    precondition(src.count == rows * cols, "shape mismatch")
    var outLen: Int = 0
    let ptr: UnsafeMutablePointer<UInt8>? = src.withUnsafeBufferPointer { srcBuf in
        ane_bridge_build_weight_blob(srcBuf.baseAddress, Int32(rows), Int32(cols), &outLen)
    }
    guard let buf = ptr else { return Data() }
    let data = Data(bytes: buf, count: outLen)
    ane_bridge_free_blob(buf)
    return data
}
