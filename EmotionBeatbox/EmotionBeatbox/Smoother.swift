/// Majority vote over the last frames that cleared the confidence
/// threshold, as in scripts/webcam_test.py. Single frames flicker even when
/// the face holds still; this is what keeps the beat from flickering too.
struct Smoother {
    /// ~0.33 s at 30 fps.
    let window = 10
    /// Settled in the webcam tests: about two thirds of frames pass, ~87% of
    /// those right. 0.7 lets only half through.
    let threshold: Float = 0.6
    private var recent: [(expression: Expression, confidence: Float)] = []

    /// The expression most of the confident recent frames agree on, or nil
    /// when none were confident, in which case the current pattern plays on.
    mutating func add(_ expression: Expression, confidence: Float) -> Expression? {
        recent.append((expression, confidence))
        if recent.count > window { recent.removeFirst() }
        let votes = Dictionary(grouping: recent.filter { $0.confidence >= threshold }, by: \.expression)
        return votes.max { $0.value.count < $1.value.count }?.key
    }

    mutating func reset() { recent.removeAll() }
}
