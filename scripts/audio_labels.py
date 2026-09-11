from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("BeatboxAudioDataset")
CLASSES = ["clap", "hihats", "kick", "snare"]


def parse_name(path):
    parts = Path(path).stem.split("-")
    label = parts[0]
    recording = parts[0] + "-" + parts[1]
    return label, recording


# .stem, not .name: with .name the original "kick-001.wav" would become its own
# recording, separate from its variants, and could land on the other side of a split.
assert parse_name("kick-001-a-1.wav") == ("kick", "kick-001")
assert parse_name("kick-001.wav") == ("kick", "kick-001")


def count(files):
    clips = Counter()
    recs = defaultdict(set)
    for f in files:
        label, recording = parse_name(f)
        clips[label] += 1
        recs[label].add(recording)
    for label in sorted(clips):
        n_recs = len(recs[label])
        print(f"  {label:8} {clips[label]:5} clip  {n_recs:3} rekaman")
    return clips, recs


if __name__ == "__main__":
    train_files = list((ROOT / "train").glob("*.wav"))
    test_files = list(ROOT.glob("*.wav"))

    print("train", len(train_files))
    train_clips, train_recs = count(train_files)
    print("test", len(test_files))
    test_clips, test_recs = count(test_files)

    # A fifth label means a filename that breaks the pattern — look at it, don't drop it.
    assert set(train_clips) == set(CLASSES), set(train_clips)
    assert set(test_clips) == set(CLASSES), set(test_clips)

    # The same recording in train and test would be a leak.
    all_train = set().union(*train_recs.values())
    all_test = set().union(*test_recs.values())
    overlap = all_train & all_test
    print("rekaman di train DAN test:", overlap or "nol")
    assert not overlap
