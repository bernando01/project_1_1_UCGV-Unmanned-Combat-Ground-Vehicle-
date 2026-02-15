# vision_detect.py
import cv2
import numpy as np
from picamera2 import Picamera2

# tflite_runtime 우선 사용, 없으면 tensorflow.lite 사용
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

MODEL_PATH = "best2-fp16_2cls_20000_v5s_320px_50epo+hyp+blur.tflite"

# 2클래스 기준: 0=person, 1=soldier
CLASS_ID_TARGET = 1

# YOLO 후처리 (NMS 없음, 전부 그림) 
def yolo_postprocess_no_nms(pred, orig_w, orig_h, conf_thres=0.2):
    """
    pred: (1, N, K)
      각 행: [xc, yc, w, h, obj, cls0, cls1, ...] 라고 가정
      xc, yc, w, h는 0~1 정규화라고 가정
    """
    p = pred[0]  # (N, K)
    if p.ndim != 2 or p.shape[1] < 6:
        return []

    boxes_xywh = p[:, :4]        # [xc, yc, w, h]
    obj = p[:, 4]                # objectness
    cls_probs = p[:, 5:]         # class probs
    cls_ids = cls_probs.argmax(axis=1)
    cls_scores = cls_probs.max(axis=1)
    scores = obj * cls_scores    # 최종 score

    results = []
    for i in range(len(p)):
        score = float(scores[i])
        if score < conf_thres:
            continue

        xc, yc, w, h = boxes_xywh[i]

        # 0~1 정규화 기준이라고 보고, 원본 좌표로 변환
        x1 = (xc - w / 2.0) * orig_w
        y1 = (yc - h / 2.0) * orig_h
        x2 = (xc + w / 2.0) * orig_w
        y2 = (yc + h / 2.0) * orig_h

        x1 = max(0, min(int(x1), orig_w - 1))
        y1 = max(0, min(int(y1), orig_h - 1))
        x2 = max(0, min(int(x2), orig_w - 1))
        y2 = max(0, min(int(y2), orig_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cid = int(cls_ids[i])
        results.append(([x1, y1, x2, y2], score, cid))

    return results


class VisionTracker:
    def __init__(self, model_path=MODEL_PATH, num_threads=4):
        # PiCamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480)}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # TFLite
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

        _, self.H, self.W, self.C = self.inp["shape"] 

        self.frame_idx = 0
        self.last = (None, 0.0, 0.0, 0.0)  # (bbox, score, err_x, err_y)

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass

    def step(self, infer_every=1):
        """
        반환:
          frame_rgb, bbox, score, err_x, err_y
          - bbox: (x1,y1,x2,y2) or None
          - err_x, err_y: 화면 중심 대비 물체 중심의 픽셀 오차 (cx-cx0, cy-cy0)
        """
        self.frame_idx += 1

        # PiCamera2는 RGB로 프레임을 줌 
        frame_rgb_full = self.picam2.capture_array()  # (H,W,3) RGB
        frame = cv2.cvtColor(frame_rgb_full, cv2.COLOR_RGB2BGR)  # 표시/그리기용 BGR

        h, w = frame.shape[:2]

        if self.frame_idx % infer_every == 0:
            orig_h, orig_w = frame.shape[:2]

            # 전처리: 320으로 축소 + RGB + 정규화
            frame_resized = cv2.resize(frame, (self.W, self.H))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            input_data = frame_rgb.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)  # (1, H, W, 3)

            # 모델 실행
            self.interpreter.set_tensor(self.inp['index'], input_data)
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.out['index'])  # (1, N, K)

            # 후처리: NMS 없이 나오는 박스 전부 사용 
            detections = yolo_postprocess_no_nms(
                pred, orig_w, orig_h,
                conf_thres=0.2
            )

            # soldier(클래스 1)만 남기고, "가장 왼쪽" 1개만 남기기
            detections = [d for d in detections if d[2] == CLASS_ID_TARGET]
            if detections:
                detections.sort(key=lambda d: d[0][0])  # x1 기준(가장 왼쪽)
                detections = [detections[0]]

            # 여기부터 vision_detect 접합(오차 계산/캐시)
            if detections:
                (x1, y1, x2, y2), sc, cid = detections[0]

                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                err_x = cx - (w * 0.5)
                err_y = cy - (h * 0.5)

                self.last = ((x1, y1, x2, y2), sc, err_x, err_y)
            else:
                self.last = (None, 0.0, 0.0, 0.0)

        bbox, sc, err_x, err_y = self.last
        return frame, bbox, sc, err_x, err_y


if __name__ == "__main__":
    vt = VisionTracker(model_path=MODEL_PATH, num_threads=4)
    try:
        while True:
            _, bbox, sc, err_x, err_y = vt.step(infer_every=1)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                print(f"Detected: soldier, score={sc:.2f}, box=({x1},{y1})-({x2},{y2}), err=({err_x:.1f},{err_y:.1f})")
    except KeyboardInterrupt:
        pass
    finally:
        vt.close()
