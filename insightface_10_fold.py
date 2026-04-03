import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis


# =========================
# 1. 配置路径
# =========================
LFW_ROOT = r"/mnt/e/Desktop/Job/Intern/lfw_funneled"
PAIRS_DIR = r"/mnt/e/Desktop/Job/Intern/lfw_funneled"   # pairs_01.txt ~ pairs_10.txt 所在目录
OUTPUT_CSV = "lfw_10fold_results.csv"

# 是否使用 GPU
CTX_ID = 0      # 有 GPU 用 0
# CTX_ID = -1   # 没 GPU 用 -1

DET_SIZE = (640, 640)

PAIR_FILES = [os.path.join(PAIRS_DIR, f"pairs_{i:02d}.txt") for i in range(1, 11)]


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

    # 多张脸时取最大脸
    def face_area(face):
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)

    face = max(faces, key=face_area)
    return face.embedding


def read_pairs_fold_file(pairs_file, lfw_root):
    """
    pairs_01.txt ~ pairs_10.txt:
    每4行一组：
      第1,2行 -> 正样本(label=1)
      第3,4行 -> 负样本(label=0)
    """
    with open(pairs_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 4 != 0:
        raise ValueError(f"{pairs_file} 的非空行数不是4的倍数，请检查格式。")

    pairs = []
    for i in range(0, len(lines), 4):
        pos1 = os.path.join(lfw_root, lines[i])
        pos2 = os.path.join(lfw_root, lines[i + 1])
        neg1 = os.path.join(lfw_root, lines[i + 2])
        neg2 = os.path.join(lfw_root, lines[i + 3])

        pairs.append((pos1, pos2, 1))
        pairs.append((neg1, neg2, 0))

    return pairs


def build_all_folds(pair_files, lfw_root):
    folds = []
    for pf in pair_files:
        pairs = read_pairs_fold_file(pf, lfw_root)
        folds.append(pairs)
    return folds


def collect_all_unique_images(folds):
    all_imgs = set()
    for fold in folds:
        for img1, img2, _ in fold:
            all_imgs.add(img1)
            all_imgs.add(img2)
    return sorted(list(all_imgs))


def compute_all_embeddings(image_paths):
    embedding_cache = {}
    failed_images = set()

    for img_path in tqdm(image_paths, desc="Extract embeddings", ncols=100):
        try:
            emb = get_embedding(img_path)
            if emb is None:
                failed_images.add(img_path)
            else:
                embedding_cache[img_path] = emb
        except Exception:
            failed_images.add(img_path)

    return embedding_cache, failed_images


def pairs_to_scores_labels(pairs, embedding_cache, failed_images):
    scores = []
    labels = []
    skipped = 0

    for img1, img2, label in tqdm(pairs, desc="Convert pairs to scores", leave=False, ncols=100):
        if img1 in failed_images or img2 in failed_images:
            skipped += 1
            continue
        if img1 not in embedding_cache or img2 not in embedding_cache:
            skipped += 1
            continue

        emb1 = embedding_cache[img1]
        emb2 = embedding_cache[img2]
        sim = cosine_similarity(emb1, emb2)

        scores.append(sim)
        labels.append(label)

    return np.array(scores), np.array(labels), skipped


def find_best_threshold(scores, labels, thresholds):
    best_acc = -1.0
    best_thresh = thresholds[0]

    for th in tqdm(thresholds, desc="Search threshold", leave=False, ncols=100):
        preds = (scores > th).astype(np.int32)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = th

    return best_thresh, best_acc


def eval_with_threshold(scores, labels, threshold):
    preds = (scores > threshold).astype(np.int32)
    acc = (preds == labels).mean()
    return acc


def save_results_to_csv(csv_path, fold_rows, summary_row):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "fold",
            "train_best_threshold",
            "train_best_accuracy",
            "test_accuracy",
            "usable_pairs",
            "skipped_pairs"
        ])

        for row in fold_rows:
            writer.writerow([
                row["fold"],
                f'{row["train_best_threshold"]:.4f}',
                f'{row["train_best_accuracy"]:.4f}',
                f'{row["test_accuracy"]:.4f}',
                row["usable_pairs"],
                row["skipped_pairs"]
            ])

        writer.writerow([])
        writer.writerow(["summary"])
        writer.writerow(["mean_accuracy", f'{summary_row["mean_accuracy"]:.4f}'])
        writer.writerow(["std_accuracy", f'{summary_row["std_accuracy"]:.4f}'])
        writer.writerow(["mean_threshold", f'{summary_row["mean_threshold"]:.4f}'])
        writer.writerow(["total_failed_images", summary_row["total_failed_images"]])
        writer.writerow(["total_unique_images", summary_row["total_unique_images"]])
        writer.writerow(["total_skipped_pairs", summary_row["total_skipped_pairs"]])


# =========================
# 4. 读取 10 折
# =========================
print("Loading folds...")
folds = build_all_folds(PAIR_FILES, LFW_ROOT)
for i, fold in enumerate(folds, 1):
    print(f"Fold {i}: {len(fold)} pairs")

# =========================
# 5. 一次性提取所有图片 embedding
# =========================
all_images = collect_all_unique_images(folds)
print(f"Total unique images: {len(all_images)}")

embedding_cache, failed_images = compute_all_embeddings(all_images)

print(f"Valid embeddings: {len(embedding_cache)}")
print(f"Failed images: {len(failed_images)}")

# =========================
# 6. 每折转成 scores / labels
# =========================
fold_scores_labels = []
total_skipped = 0

print("\nConverting all folds to scores...")
for i, fold in enumerate(folds, 1):
    print(f"Processing Fold {i} ...")
    scores, labels, skipped = pairs_to_scores_labels(fold, embedding_cache, failed_images)
    fold_scores_labels.append((scores, labels, skipped))
    total_skipped += skipped
    print(f"Fold {i}: usable={len(scores)}, skipped={skipped}")

print(f"Total skipped pairs: {total_skipped}")

# =========================
# 7. 10-fold evaluation
# =========================
thresholds = np.arange(-1.0, 1.0001, 0.001)

fold_accuracies = []
fold_thresholds = []
fold_rows = []

print("\nRunning 10-fold evaluation...")
for test_idx in tqdm(range(10), desc="10-fold evaluation", ncols=100):
    train_scores_list = []
    train_labels_list = []

    for i in range(10):
        if i == test_idx:
            continue
        s, l, _ = fold_scores_labels[i]
        if len(s) > 0:
            train_scores_list.append(s)
            train_labels_list.append(l)

    test_scores, test_labels, test_skipped = fold_scores_labels[test_idx]

    if len(test_scores) == 0:
        print(f"[WARN] Fold {test_idx + 1} has no usable pairs, skip.")
        continue

    train_scores = np.concatenate(train_scores_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    best_thresh, train_best_acc = find_best_threshold(train_scores, train_labels, thresholds)
    test_acc = eval_with_threshold(test_scores, test_labels, best_thresh)

    fold_thresholds.append(best_thresh)
    fold_accuracies.append(test_acc)

    fold_rows.append({
        "fold": test_idx + 1,
        "train_best_threshold": best_thresh,
        "train_best_accuracy": train_best_acc,
        "test_accuracy": test_acc,
        "usable_pairs": len(test_scores),
        "skipped_pairs": test_skipped,
    })

# =========================
# 8. 输出最终结果
# =========================
if len(fold_accuracies) > 0:
    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_thresh = float(np.mean(fold_thresholds))

    print("\n========== Final 10-Fold Result ==========")
    print(f"Fold accuracies : {[round(x, 4) for x in fold_accuracies]}")
    print(f"Mean accuracy   : {mean_acc:.4f}")
    print(f"Std accuracy    : {std_acc:.4f}")
    print(f"Mean threshold  : {mean_thresh:.4f}")

    summary_row = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_threshold": mean_thresh,
        "total_failed_images": len(failed_images),
        "total_unique_images": len(all_images),
        "total_skipped_pairs": total_skipped,
    }

    save_results_to_csv(OUTPUT_CSV, fold_rows, summary_row)
    print(f"\nResults saved to: {OUTPUT_CSV}")
else:
    print("No valid fold results.")