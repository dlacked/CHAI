"""
YOLOv8 기반 치아 세그멘테이션 및 분류 파이프라인.
구강 이미지에 대해 세그멘테이션 및 치아 번호를 예측합니다.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CROP_PAD,
    get_label_map,
    OralType,
    PathConfig
)

PathConfig.create_dir()


class chiaAIgent:
    """
    구강 이미지에 대한 세그멘테이션 및 치아 번호 분류 파이프라인.
    """

    def __init__(self, oral_type: OralType):
        self.oral_type = oral_type
        self.seg_model: Optional[YOLO] = None
        self.label_map = get_label_map(self.oral_type)
        self._processing_times: List[float] = []

    def set_seg_model(self, path: str | os.PathLike) -> None:
        """세그멘테이션 모델 가중치 경로 설정."""
        self.seg_model = YOLO(path)

    def run_seg(self, source: str | os.PathLike, device: int = 0):
        """세그멘테이션 및 분류 예측 실행."""
        if self.seg_model is None:
            raise RuntimeError("세그멘테이션 모델이 로드되지 않았습니다. set_seg_model()을 먼저 호출하세요.")
        return self.seg_model.predict(source=source, save=False, device=device)

    def run_pipeline(self, seg_results, save_dir: Optional[os.PathLike] = None) -> None:
        """
        세그멘테이션 및 분류 예측 결과에 대해 시각화(치아 번호 표시)를 수행하고 저장합니다.
        """
        self._processing_times = []
        save_path = Path(save_dir or PathConfig.PREDICT_DIR) / self.oral_type
        save_path.mkdir(parents=True, exist_ok=True)

        for result in seg_results:
            start_time = datetime.now() 
            img_bgr = result.orig_img
            h, w = img_bgr.shape[:2] 

            if result.masks is None:
                continue

            for mask_coords, box in zip(result.masks.xy, result.boxes):
                # YOLOv8 모델이 직접 분류한 클래스 인덱스 (0~11)
                pred_t_idx = int(box.cls[0].item())
                tooth_number = self.label_map.get(pred_t_idx, 'Unknown')

                x1 = float(np.min(mask_coords[:, 0]))
                x2 = float(np.max(mask_coords[:, 0]))
                y1 = float(np.min(mask_coords[:, 1]))
                y2 = float(np.max(mask_coords[:, 1]))
                x1 = int(max(0, x1 - CROP_PAD))
                y1 = int(max(0, y1 - CROP_PAD))
                x2 = int(min(w, x2 + CROP_PAD))
                y2 = int(min(h, y2 + CROP_PAD))

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 6)
                cv2.putText(
                    img_bgr, f"{tooth_number}", (x1 + 10, y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6,
                )

            stem = Path(result.path).stem
            out_file = save_path / f"{stem}.png"
            cv2.imwrite(str(out_file), img_bgr)
            elapsed = (datetime.now() - start_time).total_seconds()
            self._processing_times.append(elapsed)
            print(f"Saved: {out_file} | {elapsed:.2f}s")


if __name__ == "__main__":
    oral_type = input('lower/upper:')

    cAI = chiaAIgent(oral_type)
    cAI.set_seg_model(f'./runs/segment/{oral_type}/weights/best.pt')

    seg_predicted = cAI.run_seg(f'./dataset/val/images/{oral_type}')
    cAI.run_pipeline(seg_predicted)


