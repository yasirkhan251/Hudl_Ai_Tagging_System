from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "volleyball_dataset" / "data.yaml"
BASE_MODEL = ROOT / "models" / "yolov8n.pt"
OUTPUT = ROOT / "models" / "volleyball.pt"


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATASET}. "
            "Create the dataset folders and label the volleyball first."
        )

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATASET),
        epochs=60,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=str(ROOT / "runs" / "volleyball"),
        name="yolov8n_volleyball",
        pretrained=True,
        patience=15,
        cache=False,
    )

    best = ROOT / "runs" / "volleyball" / "yolov8n_volleyball" / "weights" / "best.pt"
    if best.exists():
        import shutil
        shutil.copy2(best, OUTPUT)
        print(f"Custom volleyball model saved to: {OUTPUT}")
    else:
        print(f"Training finished, but best.pt was not found at: {best}")


if __name__ == "__main__":
    main()
