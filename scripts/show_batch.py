import csv
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CSV = Path("data/splits/fer2013_4class.csv")

CLASSES = ["angry", "happy", "neutral", "surprise"]
TRANSFORM = transforms.ToTensor()  # convert PIL image to torch tensor, scale to [0,1]


def load_rows(split):
    """Return a list of (path, label) for the given split."""
    with CSV.open() as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader if row["split"] == split]


class FER2013(Dataset):
    def __init__(self, split):
        self.rows = load_rows(split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = TRANSFORM(Image.open(row["path"]))
        label = int(row["label"])
        return image, label

if __name__ == "__main__":
    ds = FER2013("train")
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    image, label = next(iter(loader))
    print("Jumlah Sampel:", len(ds))
    print("Jumlah Batch :", len(loader))
    print("Bentuk Batch :", image.shape)
    print("Bentuk Label :", label.shape)
    print("8 Label Awal :", label[:8].tolist())
    fig, axes = plt.subplots(4, 8, figsize=(12, 7))
    for ax, img, lab in zip(axes.ravel(), image, label):
        ax.imshow(img.squeeze(), cmap="gray")  # CHW to HWC
        ax.set_title(CLASSES[lab], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("docs/batch-check.png", dpi=90)
    print("Wrote docs/batch-check.png")
