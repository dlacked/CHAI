from pathlib import Path

class PathConfig:
    ROOT: Path = Path(__file__).resolve().parent # 최상위 루트 (절대경로)

    RUNS_DIR: Path = ROOT / "YOLO" / "runs" # 추론 결과 저장 디렉터리
    YOLO_WEIGHTS: Path = ROOT / "YOLO" / "yolov8n-seg.pt" # 세그멘테이션 모델 기본 가중치
