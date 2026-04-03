import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# =========================
# 1. 配置路径
# =========================
LFW_ROOT = r"/mnt/e/Desktop/Job/Intern/lfw_funneled"              # 例如: /data/lfw
PAIRS_FILE = r"/mnt/e/Desktop/Job/Intern/lfw_funneled/pairs_01.txt"   # 例如: /data/pairs_01.txt

# 是否使用 GPU
CTX_ID = 0      # 有 GPU 用 0
# CTX_ID = -1   # 没 GPU 用 -1

DET_SIZE = (640, 640)


# =========================
# 2. 初始化 insightface
# =========================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)


# =========================
# 3. 工具函数
# =========================
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def read_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return img


def get_embedding(img_path):
    img = read_image(img_path)
    faces = app.get(img)

    if len(faces) == 0:
        return None

    # 如果检测到多张脸，取面积最大的那张
    def face_area(face):
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)

    face = max(faces, key=face_area)
    emb = face.embedding
    return emb


def read_pairs_01_style(pairs_file, lfw_root):
    """
    pairs_01.txt 这类文件每4行一组：
      1,2 行 -> 同一个人，label=1
      3,4 行 -> 不同人，label=0
    """
    with open(pairs_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 4 != 0:
        raise ValueError(f"{pairs_file} 的非空行数不是4的倍数，请检查文件格式。")

    pairs = []
    for i in range(0, len(lines), 4):
        pos1 = os.path.join(lfw_root, lines[i])
        pos2 = os.path.join(lfw_root, lines[i + 1])
        neg1 = os.path.join(lfw_root, lines[i + 2])
        neg2 = os.path.join(lfw_root, lines[i + 3])

        pairs.append((pos1, pos2, 1))
        pairs.append((neg1, neg2, 0))

    return pairs


# =========================
# 4. 读取配对
# =========================
pairs = read_pairs_01_style(PAIRS_FILE, LFW_ROOT)
print(f"Total pairs: {len(pairs)}")


# =========================
# 5. 提取所有图片特征（带缓存）
# =========================
embedding_cache = {}
failed_images = set()

for img1_path, img2_path, _ in pairs:
    for img_path in [img1_path, img2_path]:
        if img_path in embedding_cache or img_path in failed_images:
            continue

        try:
            emb = get_embedding(img_path)
            if emb is None:
                failed_images.add(img_path)
                print(f"[WARN] No face detected: {img_path}")
            else:
                embedding_cache[img_path] = emb
        except Exception as e:
            failed_images.add(img_path)
            print(f"[WARN] Failed on {img_path}: {e}")

print(f"Valid embeddings: {len(embedding_cache)}")
print(f"Failed images: {len(failed_images)}")


# =========================
# 6. 计算所有 pair 的相似度
# =========================
scores = []
labels = []

skipped = 0
for img1_path, img2_path, label in pairs:
    if img1_path in failed_images or img2_path in failed_images:
        skipped += 1
        continue

    emb1 = embedding_cache[img1_path]
    emb2 = embedding_cache[img2_path]

    sim = cosine_similarity(emb1, emb2)
    scores.append(sim)
    labels.append(label)

scores = np.array(scores)
labels = np.array(labels)

print(f"Usable pairs: {len(scores)}")
print(f"Skipped pairs: {skipped}")


# =========================
# 7. 扫阈值，找最佳准确率
# =========================
best_acc = 0.0
best_thresh = 0.0

thresholds = np.arange(-1.0, 1.0001, 0.001)

for th in thresholds:
    preds = (scores > th).astype(np.int32)
    acc = (preds == labels).mean()
    if acc > best_acc:
        best_acc = acc
        best_thresh = th

print(f"Best threshold: {best_thresh:.4f}")
print(f"Best accuracy : {best_acc:.4f}")