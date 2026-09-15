import CoreVideo
import Vision

/// Camera frame -> the 48x48 grey face the model expects.
///
/// Every step copies scripts/webcam_test.py, because a crop that differs
/// from the one the model was measured on lowers accuracy with no error:
/// Vision's face box, widened by the "fer" margin, grey with OpenCV's
/// weights, then area-averaged down to 48x48 (OpenCV's INTER_AREA).
/// scripts/convert_coreml.py saved the Python results for the golden
/// webcam frames in data/webcam/golden-48/ to check this against.
nonisolated struct FaceCropper: Sendable {
    // MARGINS["fer"] in webcam_test.py: share of the box added on top, on
    // each side and at the bottom. Vision's box cuts off the brows without it.
    static let top = 0.11, side = 0.055, bottom = 0.0
    static let size = 48

    struct Crop: Sendable {
        let rect: CGRect        // pixels, origin top-left, margin included
        let face48: [UInt8]     // 48 x 48 grey, row-major
    }

    func crop(_ buffer: CVPixelBuffer) -> Crop? {
        let request = VNDetectFaceRectanglesRequest()
        try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
        let area = { (f: VNFaceObservation) in f.boundingBox.width * f.boundingBox.height }
        guard let face = request.results?.max(by: { area($0) < area($1) }) else { return nil }

        let W = Double(CVPixelBufferGetWidth(buffer)), H = Double(CVPixelBufferGetHeight(buffer))
        let bb = face.boundingBox
        // Vision: 0-1 with the origin at the bottom left. Pixels: origin top
        // left. Truncated to whole pixels first, as face_detect.py does.
        let x = Double(Int(bb.minX * W)), y = Double(Int((1 - bb.maxY) * H))
        let w = Double(Int(bb.width * W)), h = Double(Int(bb.height * H))
        let x0 = max(0, Int(x - Self.side * w)), y0 = max(0, Int(y - Self.top * h))
        let x1 = min(Int(W), Int(x + w + Self.side * w)), y1 = min(Int(H), Int(y + h + Self.bottom * h))
        guard x1 - x0 >= 20, y1 - y0 >= 20 else { return nil }

        let grey = greyscale(buffer, x0: x0, y0: y0, x1: x1, y1: y1)
        return Crop(rect: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0),
                    face48: Self.areaResize(grey, width: x1 - x0, height: y1 - y0))
    }

    /// OpenCV's BGR2GRAY in its fixed-point form: 0.299 R + 0.587 G + 0.114 B.
    /// Not private: the golden-frame check calls it with Python's own box.
    func greyscale(_ buffer: CVPixelBuffer, x0: Int, y0: Int, x1: Int, y1: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else { return [] }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var out = [UInt8](repeating: 0, count: (x1 - x0) * (y1 - y0))
        var i = 0
        for row in y0..<y1 {
            let p = base + row * stride
            for col in x0..<x1 {
                let b = Int(p[col * 4]), g = Int(p[col * 4 + 1]), r = Int(p[col * 4 + 2])
                out[i] = UInt8((b * 1868 + g * 9617 + r * 4899 + 8192) >> 14)
                i += 1
            }
        }
        return out
    }

    /// Area-averaging downscale (OpenCV INTER_AREA): each output pixel is the
    /// mean of the input area it covers, partial pixels weighted by overlap.
    /// Separable, so rows first, then columns.
    static func areaResize(_ src: [UInt8], width: Int, height: Int, to n: Int = size) -> [UInt8] {
        func weights(_ length: Int) -> [[(index: Int, weight: Double)]] {
            let scale = Double(length) / Double(n)
            return (0..<n).map { d in
                let a = Double(d) * scale, b = a + scale
                var taps: [(index: Int, weight: Double)] = []
                var s = Int(a)
                while Double(s) < b && s < length {
                    let overlap = min(b, Double(s + 1)) - max(a, Double(s))
                    if overlap > 0 { taps.append((s, overlap / scale)) }
                    s += 1
                }
                return taps
            }
        }
        let wx = weights(width), wy = weights(height)
        var rows = [Double](repeating: 0, count: height * n)
        for y in 0..<height {
            for (dx, taps) in wx.enumerated() {
                rows[y * n + dx] = taps.reduce(0) { $0 + Double(src[y * width + $1.index]) * $1.weight }
            }
        }
        var out = [UInt8](repeating: 0, count: n * n)
        for (dy, taps) in wy.enumerated() {
            for dx in 0..<n {
                let v = taps.reduce(0) { $0 + rows[$1.index * n + dx] * $1.weight }
                out[dy * n + dx] = UInt8(max(0, min(255, v.rounded())))
            }
        }
        return out
    }
}
