import AVFoundation
import SwiftUI

struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        ZStack(alignment: .topLeading) {
            CameraPreview(session: model.camera.session)
                .ignoresSafeArea()
            panel
                .padding(16)
                .frame(width: 300, alignment: .leading)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                .padding()
        }
        .frame(minWidth: 800, minHeight: 500)
        .task { await model.start() }
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !model.status.isEmpty {
                Text(model.status).font(.callout).foregroundStyle(.secondary)
            }
            Text(model.playing?.emoji ?? "🎧").font(.system(size: 64))
            Text(model.playing.map { "Beat: \($0.rawValue)" } ?? "Tunjukkan ekspresimu")
                .font(.title2.bold())
            if model.playing != nil {
                Text("variasi \(model.variation) · \(Int(model.bpm)) BPM")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Divider()
            Text(model.faceFound ? "Wajah terdeteksi" : "Wajah belum terdeteksi")
                .font(.caption).foregroundStyle(model.faceFound ? .green : .orange)
            if let points = model.landmarks {
                LandmarkThumb(points: points)
            } else if model.faceFound {
                Text("Titik wajah gak ketemu: CNN saja").font(.caption).foregroundStyle(.secondary)
            }
            ForEach(Array(Expression.modelOrder.enumerated()), id: \.offset) { i, expression in
                let p = i < model.probs.count ? Double(model.probs[i]) : 0
                HStack {
                    Text("\(expression.emoji) \(expression.rawValue)").frame(width: 100, alignment: .leading)
                    ProgressView(value: p)
                    Text(String(format: "%.2f", p)).monospacedDigit().frame(width: 36)
                }
                .font(.caption)
            }
            Text("Stabil: \(model.stable?.rawValue ?? "—")  (threshold 0,6)")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// What the models see: the 48 px crop enlarged to 192, with Vision's 76
/// points on top. The CNN reads the pixels, the Landmark MLP the points.
struct LandmarkThumb: View {
    let points: FaceLandmarks.Points

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                if let image = Self.cgImage(points.image) {
                    Image(decorative: image, scale: 1).resizable()
                }
                Canvas { context, size in
                    for p in points.display {
                        let dot = CGRect(x: p.x * size.width - 1.5, y: p.y * size.height - 1.5, width: 3, height: 3)
                        context.fill(Path(ellipseIn: dot), with: .color(.green))
                    }
                }
            }
            .frame(width: 120, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            Text("Yang dilihat model: crop 48 px diperbesar + 76 titik wajah. Gabungan: 55% piksel, 45% titik.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private static func cgImage(_ grey: [UInt8]) -> CGImage? {
        let n = FaceLandmarks.size
        guard let provider = CGDataProvider(data: Data(grey) as CFData) else { return nil }
        return CGImage(width: n, height: n, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: n,
                       space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

/// The live camera image behind the panel.
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer = layer
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
