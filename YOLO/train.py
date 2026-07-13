"""
YOLO 기반 치아 세그멘테이션 모델 로드 및 학습 (상·하악 통합 모델).
"""
import sys
from pathlib import Path

# 상위 폴더(루트 디렉터리)를 sys.path에 추가하여 config 모듈 임포트 허용
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO, settings
from config import PathConfig

def train_yolo() -> None:
    # 가중치 파일들이 프로젝트 내 YOLO 폴더에 저장되도록 설정
    settings.update({"weights_dir": str(PathConfig.ROOT / "YOLO")})

    data_yaml = PathConfig.ROOT / "YOLO" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"yaml file not exist: {data_yaml}")

    model = YOLO(str(PathConfig.YOLO_WEIGHTS))
    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        rect=True,
        batch=16,
        device=0,
        close_mosaic=0,
        project=str(PathConfig.RUNS_DIR),
        name="segment",
        exist_ok=True,
        workers=0,
    )


if __name__ == "__main__":
    train_yolo()




