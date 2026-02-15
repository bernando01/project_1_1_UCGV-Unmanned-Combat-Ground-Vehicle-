#여기서는 블러(흐림) 증강 추가

import cv2
import numpy as np
from pathlib import Path
import random
import shutil

# 원본 train 이미지/라벨 경로
img_dir = Path("C:\Users\0102k\OneDrive\바탕 화면\a\army\test\images")
lbl_dir = Path("C:\Users\0102k\OneDrive\바탕 화면\a\army\test\labels")

# 블러 이미지/라벨을 넣을 새 폴더 (작업용)
out_img_dir = Path("C:\Users\0102k\OneDrive\바탕 화면\a\army\test1\images")
out_lbl_dir = Path("C:\Users\0102k\OneDrive\바탕 화면\a\army\test1\labels")
out_img_dir.mkdir(parents=True, exist_ok=True)
out_lbl_dir.mkdir(parents=True, exist_ok=True)

# 원본도 같이 복사할 거면 True
COPY_ORIGINAL = True
# 전체 이미지 중 몇 %를 블러 버전으로 만들지 (0.3 = 30%)
BLUR_RATIO = 0.3

imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

print(f"원본 이미지 개수: {len(imgs)}")

def random_motion_kernel(ksize=5):
    """간단한 1D 모션블러 커널 (수평/수직 랜덤)"""
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    if random.random() < 0.5:
        # 수평
        kernel[ksize // 2, :] = 1.0 / ksize
    else:
        # 수직
        kernel[:, ksize // 2] = 1.0 / ksize
    return kernel

for img_path in imgs:
    stem = img_path.stem
    lbl_path = lbl_dir / f"{stem}.txt"

    # 1) 원본 그대로 복사 (선택)
    if COPY_ORIGINAL:
        shutil.copy2(img_path, out_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)

    # 2) 일정 비율만 블러 버전 생성
    if random.random() > BLUR_RATIO:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # 블러 타입 랜덤 선택
    blur_type = random.choice(["gaussian", "motion"])

    if blur_type == "gaussian":
        # 커널 크기 3~7 랜덤 (홀수)
        k = random.choice([3, 5, 7])
        blur_img = cv2.GaussianBlur(img, (k, k), 0)
    else:
        # 모션 블러
        k = random.choice([5, 7, 9])
        kernel = random_motion_kernel(k)
        blur_img = cv2.filter2D(img, -1, kernel)

    # 블러 이미지 이름: 원본이 a.jpg면 a_blur1.jpg
    blur_name = f"{stem}_blur.jpg"
    blur_img_path = out_img_dir / blur_name
    cv2.imwrite(str(blur_img_path), blur_img)

    # 라벨은 좌표 변화 없으니 그대로 복사
    if lbl_path.exists():
        shutil.copy2(lbl_path, out_lbl_dir / f"{stem}_blur.txt")

print("블러 증강 완료!")
