import numpy as np
import cv2 
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite  

# 클래스 이름 및 색상 정의
CLASS_NAMES = ["person", "soldier"]  # 0: person, 1: soldier
CLASS_COLORS = [
    (0, 0, 255),    # person: red box
    (0, 255, 255),  # soldier: yellow box
]

# 카메라 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (224, 224) 
picam2.preview_configuration.main.format = "RGB888" 
picam2.configure("preview")
picam2.start() 

# TFLite Interpreter 초기화
interpreter = tflite.Interpreter(model_path="model.tflite", num_threads=4)  
interpreter.allocate_tensors()  

inp = interpreter.get_input_details()[0] 
out = interpreter.get_output_details()[0]  

# 8비트 양자화용 스케일/제로포인트 정보
in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

# 모델 입력 텐서 검증 
in_shape = inp["shape"]
print("input shape:", in_shape)
_, H, W, C = in_shape  
assert C == 3, f"Expected 3-channel input, got {C}"
# 8비트 양자화 모델만 쓸 거라 입력 버퍼도 그 dtype으로 생성
inp_buf = np.empty((1, H, W, C), dtype=inp["dtype"])

# 전처리 함수 (UINT8 양자화 입력 전용)
def preprocess(frame_rgb):  
    frame_resized = cv2.resize(frame_rgb, (W, H))
    # 0~1 float로 변환 후 양자화 (UINT8)
    frame_f32 = frame_resized.astype(np.float32) / 255.0
    if in_scale > 0:
        # q = real / scale + zero_point
        q = frame_f32 / in_scale + in_zero
    else:
        # 방어용 (scale==0일 일은 거의 없지만)
        q = frame_resized.astype(np.float32)
    # UINT8 범위로 클립 후 저장
    inp_buf[0] = np.clip(q, 0, 255).astype(np.uint8)
    return inp_buf  

# ==========================
# 간단한 NMS 함수 (greedy)
# ==========================
def nms_xyxy(boxes, scores, iou_thres=0.45):
    """
    boxes: (N,4) numpy array, [x1,y1,x2,y2] (float, 픽셀 기준)
    scores: (N,) numpy array
    return: 남길 인덱스 리스트
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # score 높은 순

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

# 후처리 함수
def yolo_postprocess(pred, orig_w, orig_h, conf_thres=0.3, iou_thres=0.45): 
    """
    pred: (1, N, K)
    각 행: [xc, yc, w, h, obj, cls0, cls1, ...] 라고 가정.
    xc,yc,w,h 는 보통 0~1 정규화로 나온다고 보고, 자동 스케일링 후 NMS 적용.
    """
    p = pred[0]  
    if p.ndim != 2 or p.shape[1] < 6: 
        return []
    
    boxes_xywh = p[:, :4]  
    obj = p[:, 4]  
    cls_probs = p[:, 5:]  
    cls_ids = cls_probs.argmax(axis=1)  
    cls_scores = cls_probs.max(axis=1)  
    scores = obj * cls_scores  

    # 신뢰도 필터링
    mask = scores >= conf_thres 
    if not np.any(mask): 
        return []

    boxes_xywh = boxes_xywh[mask] 
    scores = scores[mask]  
    cls_ids = cls_ids[mask] 
    
    # 중심/폭/높이 → x1,y1,x2,y2 (float)
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)  # (N,4)

    # 좌표 스케일 자동 판별
    max_coord = float(np.max(xyxy)) if xyxy.size > 0 else 0.0

    # 1) 0~1 정규화
    if max_coord <= 2.0:
        xyxy[:, [0, 2]] *= float(orig_w)
        xyxy[:, [1, 3]] *= float(orig_h)
    # 2) 0~W/H 범위
    elif max_coord <= max(W, H) + 2:
        sx = float(orig_w) / float(W)
        sy = float(orig_h) / float(H)
        xyxy[:, [0, 2]] *= sx
        xyxy[:, [1, 3]] *= sy
    # 3) 이미 픽셀 기준이면 그대로

    # NMS 적용 (중복 박스 제거)
    keep = nms_xyxy(xyxy, scores, iou_thres=iou_thres)

    results = [] 
    for k in keep:
        x1, y1, x2, y2 = xyxy[k]

        # 정수화 + 클리핑
        x1 = max(0, min(int(x1), orig_w - 1))
        y1 = max(0, min(int(y1), orig_h - 1))
        x2 = max(0, min(int(x2), orig_w - 1))
        y2 = max(0, min(int(y2), orig_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        results.append(([x1, y1, x2, y2], float(scores[k]), int(cls_ids[k])))

    return results  

def main():  
    try:
        while True:  
            frame_rgb = picam2.capture_array()     
            orig_h, orig_w = frame_rgb.shape[:2] 
            
            inp_arr = preprocess(frame_rgb) 
            
            interpreter.set_tensor(inp['index'], inp_arr)  
            interpreter.invoke() 

            # 8비트 출력 → float32로 복원
            raw_pred = interpreter.get_tensor(out['index']) 
            if out_scale > 0:
                prediction = (raw_pred.astype(np.float32) - out_zero) * out_scale
            else:
                prediction = raw_pred.astype(np.float32)

            detections = yolo_postprocess(
                prediction, orig_w, orig_h,
                conf_thres=0.45, iou_thres=0.45
            ) 
       
            for (x1, y1, x2, y2), sc, cid in detections:
                cid = int(cid)
                # cid에 따라 클래스 이름과 색상 선택
                if 0 <= cid < len(CLASS_NAMES):
                    name = CLASS_NAMES[cid]
                    color = CLASS_COLORS[cid]
                else:
                    name = f"class_{cid}"
                    color = (255, 255, 255)

                print(f"Detected: {name}, score={sc:.2f}, box=({x1},{y1})-({x2},{y2})")

                # 바운딩 박스 및 라벨 그리기
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_rgb,
                    f"{name} {sc:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("TFLite Object Detection (Picamera2)", frame_rgb)  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  
    finally:
        picam2.stop()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()

