from pathlib import Path
from typing import Dict

BASE = Path.cwd() # 프로젝트 ROOT

"""구강 타입"""
OralType = str  # 'lower' | 'upper'
ORAL_TYPES = ("lower", "upper")


"""데이터 스플릿 타입"""
DataSplitType = str
DATA_SPLIT_TYPES = ("train", "val")


"""경로 클래스"""
class PathConfig:
    ROOT: Path = Path(__file__).resolve().parent.relative_to(BASE) # 최상위 루트에서 상대경로 추출

    DATASET_DIR: Path = ROOT / "dataset" # 치열궁 데이터셋 경로
    RUNS_DIR: Path = ROOT / "runs" # 추론 결과 저장 디렉터리
    SEGMENT_WEIGHTS_DIR: Path = RUNS_DIR / "segment" # Segmentation 모델 가중치 저장 디렉터리
    PREDICT_DIR: Path = ROOT / "predict" # 전체 파이프라인 예측 결과 저장 디렉터리
    
    SEGMENT_DEFAULT_WEIGHTS: str = "yolov8n-seg.pt" # 세그멘테이션 모델 기본 가중치

    @classmethod
    def create_dir(cls): # 디렉터리 없는 경우 생성
        for path in [cls.DATASET_DIR, cls.RUNS_DIR, cls.PREDICT_DIR, cls.SEGMENT_WEIGHTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_segment_weights_path(cls, oral_type: OralType) -> Path: # 치열궁 형태에 따른 모델 가중치 경로 반환
        return cls.SEGMENT_WEIGHTS_DIR / oral_type / "weights" / "best.pt"


"""치아 번호 라벨 매핑"""
LABEL_MAP_LOWER: Dict[int, int] = {
    0: 31, 1: 32, 2: 33, 3: 34, 4: 35, 5: 36,
    6: 41, 7: 42, 8: 43, 9: 44, 10: 45, 11: 46,
}
LABEL_MAP_UPPER: Dict[int, int] = {
    0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16,
    6: 21, 7: 22, 8: 23, 9: 24, 10: 25, 11: 26,
}


def get_label_map(oral_type: OralType) -> Dict[int, int]:
    """구강 타입에 해당하는 클래스 인덱스 -> 치아 번호 매핑 반환."""
    return LABEL_MAP_LOWER if oral_type == "lower" else LABEL_MAP_UPPER


"""모델 기본값"""
NUM_TOOTH_CLASSES = 12
CROP_PAD = 10

