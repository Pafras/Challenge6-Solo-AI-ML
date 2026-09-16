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

    /// One dot per eighth note; the lit one is sounding now. Dots on the
    /// beat are larger, so the bar reads as four counts.
    private var barDots: some View {
        // Redrawn on every display frame, reading the audio clock itself.
        TimelineView(.animation) { _ in
            let now = model.currentStep()
            let steps = now?.of ?? 8
            // Four beats to a bar, whatever the grid: every steps/4-th dot is
            // on a beat and drawn bigger.
            let perBeat = max(1, steps / 4)
            HStack(spacing: steps > 8 ? 3 : 6) {
                ForEach(0..<steps, id: \.self) { i in
                    let size: CGFloat = i % perBeat == 0 ? 12 : 8
                    Circle()
                        .fill(i == now?.step ? Color.accentColor : Color.secondary.opacity(0.35))
                        .frame(width: size, height: size)
                }
            }
            .frame(height: 14)
        }
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !model.status.isEmpty {
                Text(model.status).font(.callout).foregroundStyle(.secondary)
            }
            if !model.genreOptions.isEmpty {
                // The user picks the genre; the model only ever picks the expression.
                Picker("Genre", selection: $model.genre) {
                    ForEach(model.genreOptions, id: \.id) { option in
                        Text(option.label).tag(option.id)
                    }
                }
                .pickerStyle(.segmented).labelsHidden()
            }
            Text(model.playing?.emoji ?? "🎧").font(.system(size: 64))
            Text(model.playing.map { "Beat: \($0.rawValue)" } ?? "Tunjukkan ekspresimu")
                .font(.title2.bold())
            if model.playing != nil {
                Text("variasi \(model.variation)\(model.filled ? " · fill 🔥" : "") · \(Int(model.bpm)) BPM")
                    .font(.callout).foregroundStyle(.secondary)
                barDots
                if model.baseline == nil {
                    // Clear enough through the bar, and the next bar is a fill.
                    Text(String(format: "kejelasan bar ini %.2f · fill kalau ≥ %.2f", model.clarity ?? 0, model.fillAt))
                        .font(.caption).monospacedDigit()
                        .foregroundStyle((model.clarity ?? 0) >= model.fillAt ? Color.orange : Color.secondary)
                } else {
                    // Calibrated: the face's own travel picks the variation.
                    Text(String(format: "kekuatan ekspresi %.3f → variasi %@", model.strength ?? 0, model.variation))
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
                // The face has settled on another expression: say when it takes over,
                // so nobody has to count bars by ear.
                if let next = model.stable, next != model.playing {
                    Text("Berikutnya: \(next.emoji) \(next.rawValue) · di awal bar")
                        .font(.callout.bold())
                }
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
            Button(model.calibrating ? "Tahan muka datar…"
                   : model.baseline == nil ? "Kalibrasi muka datar (2 dtk)" : "Kalibrasi ulang") {
                model.calibrate()
            }
            .font(.caption)
            .disabled(model.calibrating || !model.faceFound)
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
