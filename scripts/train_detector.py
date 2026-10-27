"""Fine-tune RF-DETR on the puck / mallet labels (scripts/roboflow_pull.py).

Trains at a higher resolution than RF-DETR's default because pucks are only 15-50 px wide in
1080p frames and every frame is squashed to a square: at 576 px a far puck is ~4 px across.
Writes checkpoints and metrics to models/detector/<name>/.
Usage: python scripts/train_detector.py [--data data/labels/detector/v1] [--name v1] [--size medium] [--resolution 1280]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labels/detector/v1")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--size", default="medium", choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--resolution", type=int, default=1280, help="multiple of 64")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    import rfdetr
    model_cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
                 "base": rfdetr.RFDETRBase, "large": rfdetr.RFDETRLarge}[a.size]
    out = ROOT / "models/detector" / a.name
    out.mkdir(parents=True, exist_ok=True)
    model = model_cls(resolution=a.resolution)
    # Periodic checkpoints are 570 MB each (optimizer state included) and filled the home quota on v1;
    # keep only last.ckpt plus the small best/EMA .pth weights.
    model.train(dataset_dir=str(ROOT / a.data), output_dir=str(out), epochs=a.epochs, batch_size=a.batch,
                grad_accum_steps=a.accum, lr=a.lr, log_per_class_metrics=True, run_test=True,
                checkpoint_interval=a.epochs, progress_bar=None, tensorboard=False, seed=0)


if __name__ == "__main__":
    main()
