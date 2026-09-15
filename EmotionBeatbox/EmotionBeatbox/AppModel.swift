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
    var landmarks: FaceLandmarks.Points?
    var stable: Expression?
    var playing: Expression?
    var variation = "A"
    var bpm: Double = 0
    var step: Int?
    var stepsInBar = 8
    var clarity: Float?
    var fillAt: Float = 1
    var filled = false

    let camera = CameraManager()
    private var smoother = Smoother()
    /// A new stable expression must hold this long before it reaches the
    /// beat: the smoother alone reacts in ~0.33 s, quick enough that pulling
    /// faces for fun kept switching the pattern.
    private let holdSeconds = 0.8
    private var candidate: Expression?
    private var candidateSince = Date()
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
        // The engine's own state, for the screen. 20 times a second: a step
        // lasts 0.23 s at 130 BPM, and the dot must not skip one.
        refresh = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let beat = self.beat else { return }
                self.playing = beat.playing
                self.variation = beat.variation
                self.bpm = beat.bpm
                self.step = beat.step
                self.stepsInBar = beat.stepsInBar
                self.clarity = beat.clarity
                self.fillAt = beat.fillAt
                self.filled = beat.filled
            }
        }
    }

    private func handle(_ result: FrameProcessor.Result?) {
        faceFound = result != nil
        landmarks = result?.landmarks
        guard let result else { return }
        raw = result.expression
        confidence = result.confidence
        probs = result.probs
        beat?.observe(result.probs)
        stable = smoother.add(result.expression, confidence: result.confidence)
        // Below the threshold the smoother returns nil: keep the current
        // pattern playing rather than stopping or switching.
        guard let stable else { return }
        if stable != candidate {
            candidate = stable
            candidateSince = .now
        }
        if Date.now.timeIntervalSince(candidateSince) >= holdSeconds { beat?.target = stable }
    }
}
