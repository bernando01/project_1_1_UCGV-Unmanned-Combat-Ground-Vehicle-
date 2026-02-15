import numpy as np
import cv2
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time

# =========================
# 서보 설정 (GPIO18 사용)
# =========================
SERVO_PIN = 18          # BCM 번호 (physical pin 12)
SERVO_FREQ = 50         # 50Hz
current_angle = 90.0    # 시작 각도(정면)

def angle_to_duty(angle: float) -> float:
    # 대략 0도≈2.5%, 180도≈12.5%
    return 2.5 + (angle / 180.0) * 10.0

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
servo_pwm.start(angle_to_duty(current_angle))
time.sleep(0.3)

# =========================
# 카메라 설정
# =========================
picam2 = Picamera2()
picam2.preview_configuration.main.size = (224, 224)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# =========================
# TFLite Interpreter 초기화
# (FP16 모델이지만 입출력은 float32)
# =========================
# model.tflite <- best-fp16.tflite 복사해서 이름 맞춰두기
interpreter = tflite.Interpreter(model_path="model.tflite", num_threads=4)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

in_shape = inp["shape"]
print("input shape:", in_shape, "dtype:", inp["dtype"])
_, H, W, C = in_shape
assert C == 3, f"Expected 3-channel input, got {C}"

# FP16 TFLite는 보통 float32 입력을 받음
inp_buf = np.empty((1, H, W, C), dtype=np.float32)

# 전처리: HWC uint8 → HWC float32(0~1)
def preprocess(frame_rgb):
    frame_resized = cv2.resize(frame_rgb, (W, H))
    frame_f32 = frame_resized.astype(np.float32) / 255.0
    inp_buf[0] = frame_f32
    return inp_buf

# ==========================
# 간단한 NMS 함수 (greedy)
# ==========================
def nms_xyxy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep

# 후처리: YOLOv5 TFLite 출력 -> (box, score, class)
def yolo_postprocess(pred, orig_w, orig_h, conf_thres=0.45, iou_thres=0.45):
    p = pred[0]
    if p.ndim != 2 or p.shape[1] < 6:
        return []

    boxes_xywh = p[:, :4]          # [cx, cy, w, h] (0~1 정규화 가정)
    obj = p[:, 4]
    cls_probs = p[:, 5:]
    cls_ids = cls_probs.argmax(axis=1)
    cls_scores = cls_probs.max(axis=1)
    scores = obj * cls_scores

    mask = scores >= conf_thres
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    cx = boxes_xywh[:, 0] * orig_w
    cy = boxes_xywh[:, 1] * orig_h
    w  = boxes_xywh[:, 2] * orig_w
    h  = boxes_xywh[:, 3] * orig_h

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_xyxy(xyxy, scores, iou_thres=iou_thres)

    results = []
    for k in keep:
        xx1, yy1, xx2, yy2 = xyxy[k]

        xx1 = max(0, min(int(xx1), orig_w - 1))
        yy1 = max(0, min(int(yy1), orig_h - 1))
        xx2 = max(0, min(int(xx2), orig_w - 1))
        yy2 = max(0, min(int(yy2), orig_h - 1))

        if xx2 <= xx1 or yy2 <= yy1:
            continue

        results.append(([xx1, yy1, xx2, yy2], float(scores[k]), int(cls_ids[k])))

    return results

def main():
    global current_angle, servo_pwm

    SOLDIER_CLASS_ID = 1  # 0: person, 1: soldier
    last_update_time = time.time()

    try:
        while True:
            frame_rgb = picam2.capture_array()
            orig_h, orig_w = frame_rgb.shape[:2]
            frame_center_x = orig_w / 2.0

            inp_arr = preprocess(frame_rgb)
            interpreter.set_tensor(inp["index"], inp_arr)
            interpreter.invoke()

            prediction = interpreter.get_tensor(out["index"]).astype(np.float32)

            # soldier 인식 기준을 약간 빡세게 (conf_thres 0.55)
            detections = yolo_postprocess(
                prediction, orig_w, orig_h,
                conf_thres=0.55,
                iou_thres=0.45
            )

            # soldier만 추출
            soldier_dets = [d for d in detections if d[2] == SOLDIER_CLASS_ID]

            pos_text = "NO SOLDIER"

            if soldier_dets:
                # 중앙에 soldier!! 표시
                center_x = orig_w // 2
                center_y = orig_h // 2
                cv2.putText(
                    frame_rgb,
                    "soldier!!",
                    (center_x - 60, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # 가장 왼쪽 soldier 박스 선택
                left_box, left_score, left_cid = min(
                    soldier_dets, key=lambda d: d[0][0]
                )

                # 신뢰도가 낮으면 무시
                if left_score < 0.55:
                    pos_text = "WEAK SOLDIER"
                    # 이 경우도 서보 OFF
                    servo_pwm.ChangeDutyCycle(0)
                else:
                    x1, y1, x2, y2 = left_box
                    cx = (x1 + x2) / 2.0

                    # 프레임 중앙 대비 에러(-1 ~ 1 근처)
                    error = (cx - frame_center_x) / frame_center_x

                    # 데드존: 중앙 ±5% 안이면 안 움직임
                    dead_zone = 0.05
                    if abs(error) < dead_zone:
                        pos_text = "CENTER"
                        # 살짝 중앙에 거의 맞으면 그냥 서보 유지
                    else:
                        # 에러 크기에 비례해서 각도 변경 (비례제어)
                        k = 12.0          # 제어 강도
                        delta = k * error # error>0 → 오른쪽, error<0 → 왼쪽

                        # 한 번에 너무 크게 안 움직이도록 제한
                        max_step = 4.0
                        if delta > max_step:
                            delta = max_step
                        elif delta < -max_step:
                            delta = -max_step

                        target_angle = current_angle + delta
                        target_angle = max(30.0, min(150.0, target_angle))

                        # 각도 스무딩
                        alpha = 0.3
                        current_angle = (1 - alpha) * current_angle + alpha * target_angle

                        if error < -dead_zone:
                            pos_text = "LEFT"
                        elif error > dead_zone:
                            pos_text = "RIGHT"

                        # 너무 자주 PWM 변경하면 떨려서 0.05초 간격으로만 업데이트
                        now = time.time()
                        if now - last_update_time > 0.05:
                            servo_pwm.ChangeDutyCycle(angle_to_duty(current_angle))
                            last_update_time = now

                    print(
                        f"Soldier score={left_score:.2f}, "
                        f"angle={current_angle:.1f}, pos={pos_text}"
                    )

            else:
                # 군인 하나도 안 잡히면 → 서보 PWM OFF (힘 빼기)
                pos_text = "NO SOLDIER"
                servo_pwm.ChangeDutyCycle(0)

            # 상태 텍스트 표시
            cv2.putText(
                frame_rgb,
                pos_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("TFLite Object Detection (Picamera2)", frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        servo_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

