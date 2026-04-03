import os
import json
import glob
import numpy as np

np.int = int  # monkey patch for compatibility

import cv2
from insightface.app import FaceAnalysis


IMG_DIR = "img"
OUT_DIR = "insightface_res"

# 支持的图片格式
IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_list(x):
    if x is None:
        return None
    return np.asarray(x).tolist()


def safe_float(x):
    if x is None:
        return None
    return float(x)


def safe_int(x):
    if x is None:
        return None
    return int(x)


def gender_to_text(gender_value):
    """
    insightface 常见返回:
    0 -> female
    1 -> male
    也可能有的版本返回别的格式，所以这里保留原值并尽量解释。
    """
    if gender_value is None:
        return None
    try:
        g = int(gender_value)
        if g == 0:
            return "female"
        elif g == 1:
            return "male"
        else:
            return str(g)
    except Exception:
        return str(gender_value)


def face_to_dict(face):
    """
    将 insightface 的 Face 对象转成可序列化 JSON 的 dict
    """
    data = {}

    # 检测框
    if hasattr(face, "bbox"):
        data["bbox"] = to_list(face.bbox)

    # 5点关键点
    if hasattr(face, "kps"):
        data["kps_5"] = to_list(face.kps)

    # 106点关键点（如果有）
    if hasattr(face, "landmark_2d_106"):
        data["landmark_2d_106"] = to_list(face.landmark_2d_106)

    # 3D 68 landmarks（如果有）
    if hasattr(face, "landmark_3d_68"):
        data["landmark_3d_68"] = to_list(face.landmark_3d_68)

    # 检测分数
    if hasattr(face, "det_score"):
        data["det_score"] = safe_float(face.det_score)

    # 性别和年龄
    if hasattr(face, "gender"):
        data["gender"] = safe_int(face.gender)
        data["gender_text"] = gender_to_text(face.gender)

    if hasattr(face, "age"):
        data["age"] = safe_int(face.age)

    # 姿态（如果有）
    if hasattr(face, "pose"):
        data["pose"] = to_list(face.pose)

    # embedding（通常维度较大；如不想保存可注释掉）
    if hasattr(face, "embedding"):
        data["embedding"] = to_list(face.embedding)

    # embedding 范数（如果有）
    if hasattr(face, "normed_embedding"):
        data["normed_embedding"] = to_list(face.normed_embedding)

    return data


def collect_images(img_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
        files.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    return sorted(files)


def main():
    ensure_dir(OUT_DIR)

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image_paths = collect_images(IMG_DIR)
    print(f"Found {len(image_paths)} images in '{IMG_DIR}'")

    for img_path in image_paths:
        print(f"Processing: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Failed to read image: {img_path}")
            continue

        faces = app.get(img)
        print(f"  faces: {len(faces)}")

        # 保存结果图
        drawn = app.draw_on(img, faces)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(os.path.basename(img_path))[1]

        out_img_path = os.path.join(OUT_DIR, f"{base_name}{ext}")
        out_json_path = os.path.join(OUT_DIR, f"{base_name}.json")

        cv2.imwrite(out_img_path, drawn)

        # 保存 JSON
        result = {
            "image_name": os.path.basename(img_path),
            "image_path": img_path,
            "num_faces": len(faces),
            "faces": [face_to_dict(face) for face in faces]
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()