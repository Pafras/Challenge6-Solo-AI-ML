import CoreML
import CoreVideo
import Vision

/// The 48x48 grey crop -> Vision's 76 face points, as the Landmark MLP expects.
///
/// Every step copies scripts/extract_landmarks.py, as FaceCropper copies
/// webcam_test.py: the crop enlarged to 192 with OpenCV's bicubic (Vision
/// rarely finds a face at 48 px), VNDetectFaceLandmarksRequest, then every
/// point minus the pupils' midpoint, divided by the pupil distance.
/// scripts/convert_landmarks.py saved Python's points for the golden crops
/// in data/webcam/golden-48/landmarks.json to check this against.
nonisolated struct FaceLandmarks: Sendable {
    static let size = 192
    static let count = 76

    struct Points: Sendable {
        let features: [Float]    // 152: x0, y0, ... in pupil distances, the model's input
        let display: [CGPoint]   // 76: 0-1 inside the 192 image, origin top-left
        let image: [UInt8]       // the 192x192 grey image Vision looked at
    }

    /// nil when Vision finds no face or not 76 points: the app then keeps
    /// the CNN's answer, as fuse_landmarks.py does.
    func find(_ face48: [UInt8]) -> Points? {
        let image = Self.cubicResize(face48, from: FaceCropper.size, to: Self.size)
        guard let buffer = try? EmotionModel.pixelBuffer(image, size: Self.size) else { return nil }
        let request = VNDetectFaceLandmarksRequest()
        try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
        let area = { (f: VNFaceObservation) in f.boundingBox.width * f.boundingBox.height }
        guard let face = request.results?.filter({ $0.landmarks != nil }).max(by: { area($0) < area($1) }),
              let marks = face.landmarks, let all = marks.allPoints, all.pointCount == Self.count,
              let left = marks.leftPupil?.normalizedPoints.first,
              let right = marks.rightPupil?.normalizedPoints.first else { return nil }
        // Every region shares one frame, 0-1 inside the face box, so the
        // pupils serve directly as the reference for the 76 points.
        let cx = (left.x + right.x) / 2, cy = (left.y + right.y) / 2
        let dist = hypot(right.x - left.x, right.y - left.y)
        guard dist >= 1e-3 else { return nil }
        let pts = all.normalizedPoints, bb = face.boundingBox
        return Points(
            features: pts.flatMap { [Float(($0.x - cx) / dist), Float(($0.y - cy) / dist)] },
            // Box-relative to image-relative, and Vision's y-up to the screen's y-down.
            display: pts.map { CGPoint(x: bb.minX + $0.x * bb.width, y: 1 - (bb.minY + $0.y * bb.height)) },
            image: image)
    }

    /// How far this face sits from the user's resting face: the mean travel
    /// of the 76 points, in pupil distances (scripts/strength_thresholds.py).
    /// One number for every expression, so no per-expression geometry is
    /// needed, and the user's own flat face cancels out face shape.
    static func strength(_ features: [Float], from baseline: [Float]) -> Float? {
        guard features.count == baseline.count, !features.isEmpty else { return nil }
        var total: Float = 0
        for i in stride(from: 0, to: features.count, by: 2) {
            total += hypot(features[i] - baseline[i], features[i + 1] - baseline[i + 1])
        }
        return total / Float(features.count / 2)
    }

    /// OpenCV's INTER_CUBIC: Keys cubic with a = -0.75, pixel centres
    /// aligned, edge pixels repeated. Separable, so rows first, then columns.
    static func cubicResize(_ src: [UInt8], from n: Int, to m: Int) -> [UInt8] {
        let a = -0.75, scale = Double(n) / Double(m)
        let taps: [[(index: Int, weight: Double)]] = (0..<m).map { d in
            let f = (Double(d) + 0.5) * scale - 0.5
            let s = Int(f.rounded(.down)), t = f - Double(s)
            let w0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
            let w1 = ((a + 2) * t - (a + 3)) * t * t + 1
            let w2 = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
            return zip((s - 1)...(s + 2), [w0, w1, w2, 1 - w0 - w1 - w2]).map { (min(max($0, 0), n - 1), $1) }
        }
        var rows = [Double](repeating: 0, count: n * m)
        for y in 0..<n {
            for (dx, tp) in taps.enumerated() {
                rows[y * m + dx] = tp.reduce(0) { $0 + Double(src[y * n + $1.index]) * $1.weight }
            }
        }
        var out = [UInt8](repeating: 0, count: m * m)
        for (dy, tp) in taps.enumerated() {
            for dx in 0..<m {
                let v = tp.reduce(0) { $0 + rows[$1.index * m + dx] * $1.weight }
                out[dy * m + dx] = UInt8(max(0, min(255, v.rounded())))
            }
        }
        return out
    }
}

/// LandmarkClassifier.mlpackage, made by scripts/convert_landmarks.py. The
/// standardisation with train's mean/std and the softmax are inside it.
/// Output "probs" is in Expression.modelOrder.
nonisolated final class LandmarkModel: @unchecked Sendable {
    private let model: MLModel

    init(url: URL? = bundleURL("LandmarkClassifier", "mlmodelc")) throws {
        guard let url else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "LandmarkClassifier.mlmodelc not in the app bundle"])
        }
        model = try MLModel(contentsOf: url)
    }

    func predict(_ features: [Float]) throws -> [Float] {
        let points = try MLMultiArray(shape: [1, NSNumber(value: features.count)], dataType: .float32)
        for (i, v) in features.enumerated() { points[i] = NSNumber(value: v) }
        let input = try MLDictionaryFeatureProvider(dictionary: ["points": MLFeatureValue(multiArray: points)])
        guard let probs = try model.prediction(from: input).featureValue(for: "probs")?.multiArrayValue else {
            throw CocoaError(.coderValueNotFound)
        }
        return (0..<probs.count).map { Float(truncating: probs[$0]) }
    }
}
