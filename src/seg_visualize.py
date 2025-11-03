import cv2
import json
import numpy as np


def _get_color(label):
    """
    为每个 label 生成唯一且固定的颜色。
    """
    np.random.seed(label)  # 固定 seed 确保颜色一致性
    return (
        int(np.random.randint(0, 255)),
        int(np.random.randint(0, 255)),
        int(np.random.randint(0, 255)),
    )


def seg_visualize(image_path, label_path, bbox_path=None, alpha=0.6):
    image = cv2.imread(image_path)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    num_error = False

    if label_path.endswith(".json"):
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 获取第一个图像项（根据 JSON 结构）
        first_key = next(iter(data))
        regions = data[first_key]["regions"]
        if len(regions) != 7:
            print(f"警告: 该标注文件中包含 {len(regions)} 个圆孔区域，预期为 7 个！！！")
            num_error = True

        for idx, region in enumerate(regions, start=1):
            shape_attr = region["shape_attributes"]
            if shape_attr["name"] == "polygon":
                pts = np.array(
                    list(
                        zip(
                            shape_attr["all_points_x"],
                            shape_attr["all_points_y"],
                        )))
                cv2.fillPoly(color_mask, [pts], color=_get_color(idx))
                # 计算多边形中心
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        image,
                        str(idx),
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

    elif label_path.endswith(".png"):
        mask = cv2.imread(label_path, 0)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]  # 忽略背景 (0)
        if len(unique_labels) != 7:
            print(f"警告: 该标注文件中包含 {len(unique_labels)} 个圆孔区域，预期为 7 个！！！")
            num_error = True
        for label in unique_labels:
            color_mask[mask == label] = _get_color(label)
            # 计算该区域的中心
            ys, xs = np.where(mask == label)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                cv2.putText(
                    image,
                    str(label),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    # 叠加原图与 color_mask
    overlay = cv2.addWeighted(image[:, :, ::-1], alpha, color_mask, 1 - alpha,
                              0)

    if bbox_path:
        with open(bbox_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                class_id, center_x, center_y, width, height = map(float, parts)
                h, w = image.shape[:2]
                x1 = int((center_x - width / 2) * w)
                y1 = int((center_y - height / 2) * h)
                x2 = int((center_x + width / 2) * w)
                y2 = int((center_y + height / 2) * h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2),
                              color=(0, 255, 0),
                              thickness=2)
    if num_error:
        cv2.putText(
            overlay,
            "Warning: Number of holes != 7 !!!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "example/new.png"
    # label_path = "example/new_mask.png"
    label_path = "example/new_polygon.json"
    overlay = seg_visualize(image_path, label_path)
    plt.figure(figsize=(1200 / 100, 800 / 100))
    plt.imshow(overlay)
    plt.show()
