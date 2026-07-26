from pathlib import Path

class PathConfig:
    ROOT: Path = Path(__file__).resolve().parent # 최상위 루트 (절대경로)

    DATASET_DIR: Path = ROOT.parent / "dataset" # 치열궁 데이터셋 경로
    RUNS_DIR: Path = ROOT / "YOLO" / "runs" # 추론 결과 저장 디렉터리
    PREDICT_DIR: Path = ROOT / "YOLO" / "predict" # 전체 파이프라인 예측 결과 저장 디렉터리
    
    YOLO_WEIGHTS: Path = ROOT / "YOLO" / "yolov8n-seg.pt" # 세그멘테이션 모델 기본 가중치

    @classmethod
    def create_dir(cls): # 디렉터리 없는 경우 생성
        for path in [cls.DATASET_DIR, cls.RUNS_DIR, cls.PREDICT_DIR]:
            path.mkdir(parents=True, exist_ok=True)
