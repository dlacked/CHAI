"""
YOLO 기반 치아 세그멘테이션 모델 로드 및 학습.
"""
from pathlib import Path
from typing import Optional # None 타입 여지
from ultralytics import YOLO

from config import (
    OralType,
    PathConfig,
    ORAL_TYPES
)

class Segmentation:
    def __init__(self, oral_type: OralType):
        self.oral_type: OralType = oral_type
        self.model: Optional[YOLO] = None

    def set_model(self, path:str) -> None:
        self.model = YOLO(path)

    def train(self) -> None:
        data_yaml = Path(f"./data_{self.oral_type}.yaml")
        if not data_yaml.exists():
            raise FileNotFoundError(f"yaml file not exist: {data_yaml}")

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        self.model.train(
            data=str(data_yaml),
            epochs=40,
            imgsz=640,
            rect=True,
            batch=16,
            device=0,
            close_mosaic=0,
            project=str(PathConfig.SEGMENT_WEIGHTS_DIR),
            name=self.oral_type,
        )


if __name__ == "__main__":
    while True:
        oral_type = input("\nlower/upper: ").strip()
        if oral_type in ORAL_TYPES:
            break
        else:
            print(f"Invalid input, Please try again.")
    
    seg = Segmentation(oral_type=oral_type)
    weights_dir = PathConfig.SEGMENT_WEIGHTS_DIR / oral_type

    if (weights_dir / "weights" / "best.pt").exists():
        print(f"Trained {oral_type} Seg Model already exist.")
    else:
        print(f"Initializing {oral_type} Seg Model training.")
        seg.set_model(PathConfig.SEGMENT_DEFAULT_WEIGHTS)
        seg.train()
