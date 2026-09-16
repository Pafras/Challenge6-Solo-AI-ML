import AVFoundation

/// audio/patterns.json, the table scripts/make_samples.py renders its
/// previews from, so the app and the previews cannot disagree.
struct PatternTable: Decodable {
    struct Entry: Decodable {
        let mood: String
        let bpm: Double
        /// 2 = eighth notes (the default), 4 = sixteenths, which trap and
        /// EDM need for hat rolls.
        let stepsPerBeat: Int?
        let A: [String]
        let B: [String]
        let C: [String]

        func steps(_ variation: String) -> [String] {
            switch variation { case "B": B; case "C": C; default: A }
        }
    }

    /// Per expression, and about the face rather than the music, so every
    /// genre shares it.
    struct Face: Decodable {
        let fillAt: Float
        let strengthB: Float
        let strengthC: Float
    }

    struct Genre: Decodable {
        let label: String
        let expressions: [String: Entry]
    }

    let barsPerVariation: Int
    let order: [String]
    let samplesPerSound: Int
    let gain: [String: Float]
    let expressionSettings: [String: Face]
    let defaultGenre: String
    let genres: [String: Genre]
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
    /// How clearly the face showed the playing expression in this bar so far:
    /// the mean of its probability over the bar's frames. Clarity, not
    /// strength: the model's certainty, which is why each expression has its
    /// own fill_at in patterns.json (sad rarely tops 0.55, surprise 0.85).
    private(set) var clarity: Float?
    private(set) var fillAt: Float = 1
    /// This bar is a fill: C once, because the bar before it was clear.
    private(set) var filled = false
    private var claritySum: Float = 0
    private var clarityCount = 0
    /// Mean travel of the face from the user's resting face over this bar,
    /// once they have calibrated one. It picks the variation instead of the
    /// rotation: the face itself says how busy the beat gets.
    private(set) var strength: Float?
    private var strengthSum: Float = 0
    private var strengthCount = 0
    /// Set from the smoother; applied at the next bar.
    var target: Expression?
    /// Picked by the user, never by the model, so it adds no jitter. Like a
    /// new expression, it takes over at the next bar.
    var genre: String
    var genreOptions: [(id: String, label: String)] {
        table.genres.map { (id: $0.key, label: $0.value.label) }.sorted { $0.label < $1.label }
    }

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
    /// Recently scheduled bars, so the clock can tell which step is sounding.
    private var bars: [(start: AVAudioFramePosition, stepFrames: Double, steps: Int)] = []
    private var timer: Timer?

    init() throws {
        guard let url = bundleURL("patterns", "json") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "patterns.json not in the app bundle"])
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let table = try decoder.decode(PatternTable.self, from: Data(contentsOf: url))
        self.table = table
        genre = table.defaultGenre

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
            guard let expression = nextExpression() else {
                nextBar = horizon            // nothing to play yet
                return
            }
            scheduleBar(expression)
        }
    }

    /// A pattern plays at least this many bars before another may take over,
    /// so a face flicking between expressions cannot flip the beat mid-bar.
    /// Was 2: at 70-90 BPM that held a pattern 5-7 s, and a demo that changes
    /// face every few seconds felt slow. The 0.8 s hold in AppModel still
    /// filters flicker.
    private let minBars = 1

    private func nextExpression() -> Expression? {
        guard let playing else { return target }
        guard let target, target != playing, barsHeld >= minBars else { return playing }
        return target
    }

    /// Every frame's probabilities, in Expression.modelOrder. Only the playing
    /// expression's share counts.
    func observe(_ probs: [Float], strength travel: Float?) {
        if let travel {
            strengthSum += travel
            strengthCount += 1
            strength = strengthSum / Float(strengthCount)
        }
        guard let playing, let i = Expression.modelOrder.firstIndex(of: playing), i < probs.count else { return }
        claritySum += probs[i]
        clarityCount += 1
        clarity = claritySum / Float(clarityCount)
    }

    private func scheduleBar(_ expression: Expression) {
        // Fill: the bar ending now showed the expression clearly, so this bar
        // plays C once, then the rotation carries on. Decided once per bar,
        // never per frame, so it adds no jitter; never two bars running, or
        // C would stop sounding special.
        let fill = expression == playing && !filled
            && (clarity ?? 0) >= (table.expressionSettings[expression.rawValue]?.fillAt ?? 1)
        if expression != playing {           // a new expression starts at A
            playing = expression
            barsHeld = 0
        }
        guard let entry = table.genres[genre]?.expressions[expression.rawValue],
              let face = table.expressionSettings[expression.rawValue] else { return }
        if let travel = strength {
            // Calibrated: how far the face travelled from its resting shape
            // decides how busy this bar is. The clarity fill steps aside, so
            // only one thing ever picks the variation.
            variation = travel >= face.strengthC ? "C" : (travel >= face.strengthB ? "B" : "A")
            filled = false
        } else {
            variation = fill ? "C" : table.order[(barsHeld / table.barsPerVariation) % table.order.count]
            filled = fill
        }
        fillAt = face.fillAt
        claritySum = 0
        clarityCount = 0
        clarity = nil
        strengthSum = 0
        strengthCount = 0
        strength = nil
        bpm = entry.bpm
        let steps = entry.steps(variation)
        let stepFrames = sampleRate * 60 / entry.bpm / Double(entry.stepsPerBeat ?? 2)
        for (i, step) in steps.enumerated() {
            let at = nextBar + AVAudioFramePosition(Double(i) * stepFrames)
            for sound in step where sound != "-" { hit(String(sound), at: at) }
        }
        bars = Array((bars + [(nextBar, stepFrames, steps.count)]).suffix(4))
        nextBar += AVAudioFramePosition(Double(steps.count) * stepFrames)
        barsHeld += 1
    }

    /// The eighth note sounding now and the bar's length, for the dots on
    /// screen; nil when silent. Read straight off the audio clock on every
    /// display frame: polled through timers, the dot landed on the timers'
    /// grid instead of the beat's, 0-80 ms late, and stepped unevenly.
    /// Bars are scheduled 0.15 s ahead, so the one sounding is the latest
    /// that has already started.
    func currentStep() -> (step: Int, of: Int)? {
        let t = now
        guard let bar = bars.last(where: { $0.start <= t }) else { return nil }
        let s = Int(Double(t - bar.start) / bar.stepFrames)
        return s < bar.steps ? (s, bar.steps) : nil
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
