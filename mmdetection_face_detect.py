import os
import cv2
import numpy as np
from mmdet.apis import DetInferencer


def detect_faces_in_folder():
    input_dir = "img"
    output_dir = "mmdetection_res"
    score_thr = 0.5
    device = "cuda:0"   # 没有 GPU 就改成 "cpu"

    # 支持的图片格式
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    inferencer = DetInferencer(
        model="retinanet_r50_fpn_1x_coco",
        device=device
    )

    # 获取所有图片
    image_names = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    image_names.sort()

    if len(image_names) == 0:
        print(f"文件夹 {input_dir} 中没有找到图片。")
        return

    print(f"共找到 {len(image_names)} 张图片，开始检测...")

    for image_name in image_names:
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        # 推理
        results = inferencer(
            image_path,
            return_vis=False,
            no_save_pred=True,
            no_save_vis=True
        )

        pred = results["predictions"][0]
        bboxes = np.array(pred.get("bboxes", []), dtype=np.float32)
        scores = np.array(pred.get("scores", []), dtype=np.float32)
        labels = np.array(pred.get("labels", []), dtype=np.int32)

        # COCO 中 person 类别一般是 0
        keep = (scores >= score_thr) & (labels == 0)

        bboxes = bboxes[keep]
        scores = scores[keep]

        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 画框
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            x1, y1, x2, y2 = bbox.astype(int).tolist()

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"person {score:.2f}",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(
                f"{image_name} -> [{i}] "
                f"bbox=({x1}, {y1}, {x2}, {y2}), score={score:.3f}"
            )

        # 保存结果
        cv2.imwrite(output_path, img)
        print(f"{image_name} 检测完成，结果已保存到: {output_path}")

    print("全部图片处理完成。")


if __name__ == "__main__":
    detect_faces_in_folder()