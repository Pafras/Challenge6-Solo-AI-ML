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
    /// The user's choice, applied at the next bar. Empty until the engine loads.
    var genre = "" { didSet { beat?.genre = genre } }
    var genreOptions: [(id: String, label: String)] = []
    var clarity: Float?
    var fillAt: Float = 1
    var filled = false
    /// The user's resting face, averaged over the calibration seconds.
    /// Until it exists, variations rotate and the clarity fill applies.
    private(set) var baseline: [Float]?
    private(set) var strength: Float?
    var calibrating: Bool { calibrateUntil != nil }
    private var calibrateUntil: Date?
    private var calibrationFrames: [[Float]] = []

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
            genreOptions = beat.genreOptions
            genre = beat.genre
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
                self.clarity = beat.clarity
                self.strength = beat.strength
                self.fillAt = beat.fillAt
                self.filled = beat.filled
            }
        }
    }

    /// Hold a flat face for two seconds; the average of those frames becomes
    /// the resting face every later frame is measured against.
    func calibrate() {
        calibrationFrames.removeAll()
        calibrateUntil = .now.addingTimeInterval(2)
    }

    /// For the bar dots, which ask on every display frame.
    func currentStep() -> (step: Int, of: Int)? { beat?.currentStep() }

    private static func average(_ rows: [[Float]]) -> [Float]? {
        guard let first = rows.first else { return nil }
        var out = [Float](repeating: 0, count: first.count)
        for row in rows where row.count == out.count {
            for i in out.indices { out[i] += row[i] / Float(rows.count) }
        }
        return out
    }

    private func handle(_ result: FrameProcessor.Result?) {
        faceFound = result != nil
        landmarks = result?.landmarks
        guard let result else { return }
        raw = result.expression
        confidence = result.confidence
        probs = result.probs
        var travel: Float?
        if let points = result.landmarks {
            if let until = calibrateUntil {
                calibrationFrames.append(points.features)
                if Date.now >= until {
                    calibrateUntil = nil
                    baseline = Self.average(calibrationFrames)
                }
            }
            if let baseline { travel = FaceLandmarks.strength(points.features, from: baseline) }
        }
        beat?.observe(result.probs, strength: travel)
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
