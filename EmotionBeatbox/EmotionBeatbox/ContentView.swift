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
