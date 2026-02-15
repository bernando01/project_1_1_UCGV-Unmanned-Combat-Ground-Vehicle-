import numpy as np
import cv2 
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite  

# 카메라 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (224, 224) 
picam2.preview_configuration.main.format = "RGB888" 
picam2.configure("preview")
picam2.start() 

# TFLite Interpreter 초기화
# 모델 로드, 사용할 CPU 스레드 4개 허용
interpreter = tflite.Interpreter(model_path="model.tflite", num_threads=4)  
# 텐서 메모리 공간 할당
interpreter.allocate_tensors()  
# 입력 메타 데이터 갖고오기 -> 전처리 데이터 검증용
inp = interpreter.get_input_details()[0] 
# 출력 메타 데이터 가져오기 -> 후처리 함수에서 사용
out = interpreter.get_output_details()[0]  

# ======================
#  입력/출력 텐서 설정/버퍼
# ======================

# 입력/출력 dtype 및 양자화 정보
input_shape = inp['shape']
input_dtype = inp['dtype']              # np.float32, np.uint8, np.int8 등
in_scale, in_zero = inp['quantization'] # (scale, zero_point)

output_dtype = out['dtype']
out_scale, out_zero = out['quantization']

# 모델 입력 텐서 검증 
# YOLOv5 TFLite는 [1, H, W, C] (배치, 높이, 너비, 채널) 형태
_, H, W, C = input_shape
# 채널 수가 3이면 프로그램 실행 아니면 오류메시지 출력
assert C == 3, f"Expected 3-channel input, got {C}"  

# 모델 입력용 버퍼배열 생성 -> 전처리할 때마다 새로 만들면 메모리 낭비, 미리 만들어서 값만 변경해서 최적화
# [1, H, W, C] 형식, 입력 dtype에 맞춰 생성
inp_buf = np.empty((1, H, W, C), dtype=input_dtype) 

# 전처리 함수
# 카메라로 찍은 이미지를 모델의 입력 데이터 형식으로 변환
def preprocess(frame_rgb):  
    # 입력 크기(H, W)에 맞게 리사이즈 (카메라 해상도와 다를 수 있으니 안전하게)
    frame_resized = cv2.resize(frame_rgb, (W, H))

    # FP32/FP16 TFLite (float 입력)인 경우
    if input_dtype == np.float32:
        # 형변환 및 정규화 : uint8 -> float32, 0~255 -> 0~1
        rgb_f32 = frame_resized.astype(np.float32) / 255.0  
        # YOLOv5 TFLite는 HWC 그대로 사용 → [1, H, W, C]
        inp_buf[0] = rgb_f32

    # INT8 / UINT8 양자화 TFLite인 경우
    elif input_dtype in (np.uint8, np.int8):
        # 먼저 0~1 float로 맞추고
        rgb_f32 = frame_resized.astype(np.float32) / 255.0

        if in_scale > 0:
            # 양자화: q = real / scale + zero_point
            q = rgb_f32 / in_scale + in_zero
        else:
            # 방어적 코드 (scale이 0인 이상 케이스)
            q = rgb_f32 * 255.0

        if input_dtype == np.uint8:
            q = np.clip(q, 0, 255).astype(np.uint8)
        else:  # np.int8
            q = np.clip(q, -128, 127).astype(np.int8)

        inp_buf[0] = q

    else:
        raise TypeError(f"Unsupported input dtype: {input_dtype}")

    return inp_buf  

# 후처리 함수
# 모델 출력 → 박스/클래스/점수로 변환
def yolo_postprocess(pred, orig_w, orig_h, conf_thres=0.45, iou_thres=0.45): 
    # [N,7] 형태의 예측 배열 취득
    p = pred[0]  
    # 예측 형식이 기대와 다르면 빈 결과 반환
    if p.ndim != 2 or p.shape[1] < 6: 
        return []
    
    # 필드 분리 (좌표/스코어/클래스 확률)
    boxes_xywh = p[:, :4]      # [cx, cy, w, h] (0~1 범위라고 가정)
    obj = p[:, 4]  
    cls_probs = p[:, 5:]  
    cls_ids = cls_probs.argmax(axis=1)  
    cls_scores = cls_probs.max(axis=1)  
    scores = obj * cls_scores  

    # 신뢰도 필터링
    # 임계값 이상인 것만 남기고, 남는 후보 없으면 빈 결과 return
    mask = scores >= conf_thres 
    if not np.any(mask): 
        return []
    boxes_xywh = boxes_xywh[mask] 
    scores = scores[mask]  
    cls_ids = cls_ids[mask] 
    
    # 정규화된 좌표(0~1)를 원본 프레임 크기로 스케일
    cx = boxes_xywh[:, 0] * orig_w
    cy = boxes_xywh[:, 1] * orig_h
    w  = boxes_xywh[:, 2] * orig_w
    h  = boxes_xywh[:, 3] * orig_h

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # OpenCV NMS 입력 형태로 변환
    boxes = xyxy.round().astype(np.int32).tolist() 
    scores_list = scores.astype(float).tolist()  
    
    # NMS로 중복 박스 제거
    idxs = cv2.dnn.NMSBoxes(boxes, scores_list, conf_thres, iou_thres) 
    
    # 최종 결과 구성
    results = [] 
    if isinstance(idxs, (list, tuple)) and len(idxs) > 0:  
        for e in idxs:
            i = e[0] if hasattr(e, "__getitem__") else int(e) 
            results.append((boxes[i], scores_list[i], int(cls_ids[i])))  
    elif isinstance(idxs, np.ndarray) and idxs.size > 0: 
        for i in idxs.flatten():
            results.append((boxes[int(i)], scores_list[int(i)], int(cls_ids[int(i)])))  
    return results  

def main():  
    try:
        while True:  
            # 카메라 프레임 캡처 (RGB)
            frame_rgb = picam2.capture_array()     
            
            # 전처리 → 모델 입력
            inp_arr = preprocess(frame_rgb) 
            interpreter.set_tensor(inp['index'], inp_arr)  
            interpreter.invoke() 

            # 원시 출력 텐서
            raw_pred = interpreter.get_tensor(out['index'])

            # 출력이 양자화된 타입이면 float로 복원
            if output_dtype in (np.uint8, np.int8) and out_scale > 0:
                prediction = (raw_pred.astype(np.float32) - out_zero) * out_scale
            else:
                prediction = raw_pred.astype(np.float32)

            orig_h, orig_w = frame_rgb.shape[:2] 
            detections = yolo_postprocess(
                prediction, orig_w, orig_h,
                conf_thres=0.45, iou_thres=0.45
            ) 
       
            # ==============================
            # 가장 왼쪽 박스 중심에만 초록 점
            # ==============================
            if detections:  # 박스가 하나 이상 있을 때만
                # detections 원소: ((x1, y1, x2, y2), score, cid)
                left_box, left_score, left_cid = min(
                    detections, key=lambda d: d[0][0]  # x1이 가장 작은 박스
                )
                x1, y1, x2, y2 = left_box

                # 중심 좌표 계산
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 초록색 크게 채워진 점 (반지름 15)
                cv2.circle(frame_rgb, (cx, cy), 15, (0, 255, 0), -1)

                # 좌표/클래스 디버깅 출력
                print(f"Leftmost center: ({cx}, {cy}), score={left_score:.2f}, cls={left_cid}")

            # 박스 테두리는 그리지 않음
            # 필요하면 아래 주석 해제해서 사용
            """
            for (x1, y1, x2, y2), sc, cid in detections:
                if cid == 1:
                    color = (0, 255, 255)
                    name = "yellow"
                else:
                    color = (0, 0, 255)
                    name = "red"
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                print(f"Detected: {name}, score={sc:.2f}")
            """

            cv2.imshow("TFLite Object Detection (Picamera2)", frame_rgb)  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  
    finally:
        picam2.stop()  
        cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()

