import os
from pathlib import Path

labels_dir = Path(r"C:\Users\0102k\OneDrive\바탕 화면\dataset_2_class2\valid\labels")

TARGET_FROM = 0   # 바꾸기 전 클래스
TARGET_TO = 1     # 바꾼 후 클래스

txt_files = list(labels_dir.glob("*.txt"))
print(f"라벨 파일 개수: {len(txt_files)}")

for txt_path in txt_files:
    lines_out = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            cls_str = parts[0]

            try:
                cls_id = int(float(cls_str))
            except ValueError:
                continue

            if cls_id == TARGET_FROM:
                cls_str = str(TARGET_TO)   # '0.0'이든 뭐든 최종적으로 '1'로 저장

            # 나머지 좌표는 그대로 유지
            cx, cy, w, h = parts[1:5]
            lines_out.append(f"{cls_str} {cx} {cy} {w} {h}\n")

    # 변경된 내용 다시 쓰기
    with open(txt_path, "w", encoding="utf-8") as f:
        for l in lines_out:
            f.write(l)

print(f"완료! → 클래스 {TARGET_FROM} 이(가) 모두 {TARGET_TO} 으로 변경됨.")


TARGET_CHECK = 0.0   

txt_files = list(labels_dir.glob("*.txt"))
print(f"[INFO] 라벨 파일 개수: {len(txt_files)}")

remain_count = 0          # TARGET_CHECK가 남아있는 총 라인 수
files_with_target = []    # TARGET_CHECK가 남아있는 파일 리스트
class_set = set()         # 전체 등장한 클래스들
class_counts = {}         # 클래스별 라인 수 카운트

for txt_path in txt_files:
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            cls_str = parts[0]
            try:
                cls_val = float(cls_str)
            except ValueError:
                continue  # 이상한 줄은 스킵

            # 전체 클래스 통계 업데이트
            class_set.add(cls_val)
            class_counts[cls_val] = class_counts.get(cls_val, 0) + 1

            # 우리가 확인하고 싶은 클래스가 남아있는지 체크
            if abs(cls_val - TARGET_CHECK) < 1e-6:
                remain_count += 1
                files_with_target.append(txt_path)

print("\n[RESULT] 전체 클래스 종류:", sorted(class_set))
print("[RESULT] 클래스별 라인 수:")
for k in sorted(class_counts.keys()):
    print(f"  클래스 {k}: {class_counts[k]} 라인")

print(f"\n[CHECK] 아직 남아있는 클래스 {TARGET_CHECK} 라인 수: {remain_count}")

if remain_count == 0:
    print("OK: TARGET_CHECK 클래스는 더 이상 남아있지 않습니다.")
