

import os
from pathlib import Path

labels_dir = Path(r"C:\Users\0102k\OneDrive\바탕 화면\a\사람\사람3000_emp\Per\valid\labels")

txt_files = list(labels_dir.glob("*.txt"))
print(f"라벨 파일 개수: {len(txt_files)}")

for txt_path in txt_files:
    try:
        with open(txt_path, "w", encoding="utf-8"):
            pass
    except Exception as e:
        print(f"[ERROR] {txt_path} 처리 중 에러: {e}")

print("모든 라벨 파일 내용을 비웠습니다.")


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
                continue 

            # 전체 클래스 통계 업데이트
            class_set.add(cls_val)
            class_counts[cls_val] = class_counts.get(cls_val, 0) + 1

            # 우리가 확인하고 싶은 클래스가 남아있는지 체크
            if abs(cls_val - TARGET_CHECK) < 1e-6:
                remain_count += 1
                files_with_target.append(txt_path)
                # 같은 파일 여러 줄 있어도 파일 목록에는 한 번만 넣고 싶으면 break 써도 됨

print("\n[RESULT] 전체 클래스 종류:", sorted(class_set))
print("[RESULT] 클래스별 라인 수:")
for k in sorted(class_counts.keys()):
    print(f"  클래스 {k}: {class_counts[k]} 라인")

print(f"\n[CHECK] 아직 남아있는 클래스 {TARGET_CHECK} 라인 수: {remain_count}")

if remain_count == 0:
    print("TARGET_CHECK 클래스는 더 이상 남아있지 않습니다.")
