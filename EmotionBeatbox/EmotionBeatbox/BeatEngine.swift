import AVFoundation

/// audio/patterns.json, the table scripts/make_samples.py renders its
/// previews from, so the app and the previews cannot disagree.
struct PatternTable: Decodable {
    struct Entry: Decodable {
        let mood: String
        let bpm: Double
        let A: [String]
        let B: [String]
        let C: [String]

        func steps(_ variation: String) -> [String] {
            switch variation { case "B": B; case "C": C; default: A }
        }
    }

    let barsPerVariation: Int
    let order: [String]
    let samplesPerSound: Int
    let gain: [String: Float]
    let expressions: [String: Entry]
}

/// Sample playback on a bar clock. Not AI.
///
/// Each step of a pattern is an eighth note; a step lists the sounds that hit
/// together (K kick, H hi-hat, S snare, C clap). Bars are scheduled on the
/// audio clock a little ahead of time, never with a Timer firing each hit:
/// a Timer drifts by milliseconds and the ear hears a wobbly beat. A new
/// expression takes over at the next bar that has not been scheduled yet,
/// always from variation A; while it is held, the variations rotate
/// A, B, A, C, two bars each.
final class BeatEngine {
    private(set) var playing: Expression?
    private(set) var variation = "A"
    private(set) var bpm: Double = 0
    /// Set from the smoother; applied at the next bar.
    var target: Expression?

    private let table: PatternTable
    private let engine = AVAudioEngine()
    private let sampleRate: Double
    private static let tokens = ["K", "H", "S", "C"]
    private static let stems = ["K": "kick", "H": "hihat", "S": "snare", "C": "clap"]
    private var takes: [String: [AVAudioPCMBuffer]] = [:]
    // A few players per sound, so a hit can ring on while the next one starts.
    private var pools: [String: [AVAudioPlayerNode]] = [:]
    private var nextTake: [String: Int] = [:]
    private var nextPlayer: [String: Int] = [:]
    private var nextBar: AVAudioFramePosition = 0
    private var barsHeld = 0
    private var timer: Timer?

    init() throws {
        guard let url = bundleURL("patterns", "json") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "patterns.json not in the app bundle"])
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let table = try decoder.decode(PatternTable.self, from: Data(contentsOf: url))
        self.table = table

        // Loaded through a static helper: a closure in init may not touch
        // self before every stored property has a value.
        var loaded: [String: [AVAudioPCMBuffer]] = [:]
        for token in Self.tokens {
            let stem = Self.stems[token]!
            loaded[token] = try (1...table.samplesPerSound).map { k in try Self.load("\(stem)_\(k)") }
        }
        guard let format = loaded["K"]?.first?.format else { throw CocoaError(.coderValueNotFound) }
        takes = loaded
        sampleRate = format.sampleRate
        for token in Self.tokens {
            pools[token] = (0..<3).map { _ in
                let player = AVAudioPlayerNode()
                engine.attach(player)
                engine.connect(player, to: engine.mainMixerNode, format: format)
                player.volume = table.gain[token] ?? 1
                return player
            }
        }
    }

    private static func load(_ name: String) throws -> AVAudioPCMBuffer {
        guard let url = bundleURL(name, "wav") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "\(name).wav not in the app bundle"])
        }
        let file = try AVAudioFile(forReading: url)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buffer)
        return buffer
    }

    func start() throws {
        try engine.start()
        // Every player starts at the same instant, so sample 0 means the
        // same moment on all of them and a bar lines up across sounds.
        let start = AVAudioTime(hostTime: mach_absolute_time() + AVAudioTime.hostTime(forSeconds: 0.1))
        for player in pools.values.flatMap({ $0 }) { player.play(at: start) }
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.scheduleAhead() }
        }
    }

    func stop() {
        timer?.invalidate()
        engine.stop()
    }

    /// Where the audio clock is now, in samples since the players started.
    private var now: AVAudioFramePosition {
        guard let player = pools["K"]?.first, let node = player.lastRenderTime,
              let time = player.playerTime(forNodeTime: node) else { return 0 }
        return time.sampleTime
    }

    /// Keep about 0.15 s of beat scheduled ahead of the clock.
    private func scheduleAhead() {
        let horizon = now + AVAudioFramePosition(0.15 * sampleRate)
        if nextBar < now { nextBar = now }   // silent, or late: start from here
        while nextBar < horizon {
            guard let expression = target ?? playing else {
                nextBar = horizon            // nothing to play yet
                return
            }
            scheduleBar(expression)
        }
    }

    private func scheduleBar(_ expression: Expression) {
        if expression != playing {           // a new expression starts at A
            playing = expression
            barsHeld = 0
        }
        guard let entry = table.expressions[expression.rawValue] else { return }
        variation = table.order[(barsHeld / table.barsPerVariation) % table.order.count]
        bpm = entry.bpm
        let steps = entry.steps(variation)
        let stepFrames = sampleRate * 60 / entry.bpm / 2
        for (i, step) in steps.enumerated() {
            let at = nextBar + AVAudioFramePosition(Double(i) * stepFrames)
            for sound in step where sound != "-" { hit(String(sound), at: at) }
        }
        nextBar += AVAudioFramePosition(Double(steps.count) * stepFrames)
        barsHeld += 1
    }

    /// Takes and players both go round-robin: neighbouring hits of one sound
    /// are never the same file, and never cut each other off.
    private func hit(_ token: String, at sample: AVAudioFramePosition) {
        guard let buffers = takes[token], let players = pools[token] else { return }
        let take = nextTake[token, default: 0], p = nextPlayer[token, default: 0]
        nextTake[token] = take + 1
        nextPlayer[token] = p + 1
        players[p % players.count].scheduleBuffer(buffers[take % buffers.count],
                                                  at: AVAudioTime(sampleTime: sample, atRate: sampleRate))
    }
}
