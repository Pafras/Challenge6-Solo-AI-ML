import Foundation

/// The five expressions, in the order of the model's output neurons.
/// Reordering this silently mislabels every prediction.
nonisolated enum Expression: String, CaseIterable, Sendable {
    case angry, happy, neutral, surprise, sad

    /// Output neuron i of EmotionClassifier is modelOrder[i]
    /// (checkpoint classes: angry, happy, neutral, surprise, sad).
    static let modelOrder: [Expression] = [.angry, .happy, .neutral, .surprise, .sad]

    var emoji: String {
        switch self {
        case .angry: "😠"
        case .happy: "😄"
        case .neutral: "😐"
        case .surprise: "😮"
        case .sad: "😢"
        }
    }
}

/// A file the app bundles. Synchronized folders may keep the Resources
/// subfolder or flatten it, so both places are tried.
nonisolated func bundleURL(_ name: String, _ ext: String) -> URL? {
    Bundle.main.url(forResource: name, withExtension: ext)
        ?? Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Resources")
}
