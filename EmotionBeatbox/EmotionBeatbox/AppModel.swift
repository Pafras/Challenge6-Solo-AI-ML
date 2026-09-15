import Foundation
import Observation

/// Camera -> crop -> model -> smoother -> beat, and what the screen shows.
@Observable
final class AppModel {
    var status = "Menyiapkan…"
    var faceFound = false
    var raw: Expression?
    var confidence: Float = 0
    var probs: [Float] = []
    var stable: Expression?
    var playing: Expression?
    var variation = "A"
    var bpm: Double = 0

    let camera = CameraManager()
    private var smoother = Smoother()
    private var beat: BeatEngine?
    private var refresh: Timer?

    func start() async {
        let processor: FrameProcessor
        do {
            processor = try FrameProcessor()
            let beat = try BeatEngine()
            try beat.start()
            self.beat = beat
        } catch {
            status = "Gagal memuat: \(error.localizedDescription)"
            return
        }
        // Runs on the camera's queue; only the result hops to the main thread.
        // ponytail: strong capture. The model lives as long as the app, so the
        // cycle through the camera never needs breaking; a weak var cannot
        // cross into the main-actor Task safely.
        camera.onFrame = { frame in
            let result = processor.process(frame)
            Task { @MainActor in self.handle(result) }
        }
        guard await camera.start() else {
            status = "Kamera tidak diizinkan. Buka System Settings > Privacy & Security > Camera."
            return
        }
        status = ""
        // The engine's own state, for the screen, a few times a second.
        refresh = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let beat = self.beat else { return }
                self.playing = beat.playing
                self.variation = beat.variation
                self.bpm = beat.bpm
            }
        }
    }

    private func handle(_ result: FrameProcessor.Result?) {
        faceFound = result != nil
        guard let result else { return }
        raw = result.expression
        confidence = result.confidence
        probs = result.probs
        stable = smoother.add(result.expression, confidence: result.confidence)
        // Below the threshold the smoother returns nil: keep the current
        // pattern playing rather than stopping or switching.
        if let stable { beat?.target = stable }
    }
}
