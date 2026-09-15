import CoreML
import CoreVideo

/// EmotionClassifier.mlpackage, made by scripts/convert_coreml.py. It takes
/// the 48x48 grey crop and does the rest itself: up to 224, three channels,
/// ImageNet normalisation, softmax. Output "probs" is in Expression.modelOrder.
nonisolated final class EmotionModel: @unchecked Sendable {
    private let model: MLModel

    init() throws {
        guard let url = bundleURL("EmotionClassifier", "mlmodelc") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "EmotionClassifier.mlmodelc not in the app bundle"])
        }
        model = try MLModel(contentsOf: url)
    }

    func predict(_ face48: [UInt8]) throws -> [Float] {
        let input = try MLDictionaryFeatureProvider(dictionary: ["face48": MLFeatureValue(pixelBuffer: try Self.pixelBuffer(face48))])
        guard let probs = try model.prediction(from: input).featureValue(for: "probs")?.multiArrayValue else {
            throw CocoaError(.coderValueNotFound)
        }
        return (0..<probs.count).map { Float(truncating: probs[$0]) }
    }

    /// One-channel 8-bit buffer: Core ML scales 0-255 to 0-1 itself, as
    /// ToTensor did in training. FaceLandmarks hands Vision its 192 image
    /// the same way.
    static func pixelBuffer(_ pixels: [UInt8], size n: Int = FaceCropper.size) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, n, n, kCVPixelFormatType_OneComponent8, nil, &buffer)
        guard let buffer else { throw CocoaError(.coderInvalidValue) }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for row in 0..<n {
            pixels.withUnsafeBufferPointer { src in
                (base + row * stride).update(from: src.baseAddress! + row * n, count: n)
            }
        }
        return buffer
    }
}

/// Camera frame -> crop -> CNN + Landmark MLP, mixed, run on the camera's
/// queue so the main thread only ever sees the result.
nonisolated final class FrameProcessor: @unchecked Sendable {
    struct Result: Sendable {
        let expression: Expression
        let confidence: Float
        let probs: [Float]
        let landmarks: FaceLandmarks.Points?   // nil: no points, probs are the CNN's alone
    }

    /// The CNN's share of the mix, chosen on FER valid in run 20
    /// (scripts/fuse_landmarks.py) and confirmed live in webcam test #4.
    static let alpha: Float = 0.55

    private let cropper = FaceCropper()
    private let finder = FaceLandmarks()
    private let model: EmotionModel
    private let landmarkModel: LandmarkModel

    init() throws {
        model = try EmotionModel()
        landmarkModel = try LandmarkModel()
    }

    /// nil when no face is found.
    func process(_ frame: CVPixelBuffer) -> Result? {
        guard let crop = cropper.crop(frame), let cnn = try? model.predict(crop.face48) else { return nil }
        let points = finder.find(crop.face48)
        var probs = cnn
        if let points, let lm = try? landmarkModel.predict(points.features) {
            probs = zip(cnn, lm).map { Self.alpha * $0 + (1 - Self.alpha) * $1 }
        }
        guard let best = probs.indices.max(by: { probs[$0] < probs[$1] }) else { return nil }
        return Result(expression: Expression.modelOrder[best], confidence: probs[best], probs: probs,
                      landmarks: points)
    }
}
