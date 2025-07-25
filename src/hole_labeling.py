import cv2
import glob
import numpy as np
import json
import os.path as osp
from .label_optimize import optimize
from .seg_visualize import seg_visualize


class CircleDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.polygon_path = osp.splitext(image_path)[0] + "_polygon.json"
        self.mask_path = osp.splitext(image_path)[0] + "_mask.png"
        self.original = cv2.imread(image_path)
        self.drag_start = None
        self.selected_ellipse = []
        self.flag = "canny"
        self.roi_counter = 0

        # 缩放相关参数
        self.display_size_main = (1200, 800)
        self.display_size_roi = (200, 200)
        self.scale_main = min(
            self.display_size_main[0] / self.original.shape[1],
            self.display_size_main[1] / self.original.shape[0],
        )

        self.win_name = "Circle Detection"
        self.roi_win_name = "ROI"
        
        self.current_roi_image = None
        self.canny_points = None
        self.selected_points = []
    def mouse_handler(self, event, x, y, flags, param):
        # 缩放还原为原图坐标
        x = int(x / self.scale_main)
        y = int(y / self.scale_main)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if self.drag_start:
                self.draw_preview_rect(self.drag_start, (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_start:
                x1, y1 = self.drag_start
                x2, y2 = x, y
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                if w > 5 and h > 5:
                    self.process_rectangle(x, y, w, h)

    def roi_mouse_handler(self, event, x, y, flags, param):
        x = int(x / self.scale_roi)
        y = int(y / self.scale_roi)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.flag == "canny":
                if self.canny_points is not None and len(self.canny_points) > 0:
                    distances = np.linalg.norm(
                        self.canny_points - np.array([x, y]), axis=1
                    )
                    nearest_index = np.argmin(distances)
                    nearest_point = tuple(self.canny_points[nearest_index])
                    self.selected_points.append(nearest_point)
            elif self.flag == "raw":
                self.selected_points.append((x, y))

            if self.selected_points:
                temp = self.current_roi_image.copy()
                for pt in self.selected_points:
                    pt_disp = (
                        int(pt[0] * self.scale_roi),
                        int(pt[1] * self.scale_roi),
                    )
                    cv2.circle(temp, pt_disp, 3, (0, 0, 255), -1)
                cv2.imshow(self.roi_win_name, temp)

        elif event == cv2.EVENT_MBUTTONDOWN:
            if self.selected_points:
                self.selected_points.pop()
                temp = self.current_roi_image.copy()
                for pt in self.selected_points:
                    pt_disp = (
                        int(pt[0] * self.scale_roi),
                        int(pt[1] * self.scale_roi),
                    )
                    cv2.circle(temp, pt_disp, 3, (0, 0, 255), -1)
                cv2.imshow(self.roi_win_name, temp)

    def draw_preview_rect(self, start, end):
        temp = self.get_scaled_main_image()
        cv2.rectangle(
            temp,
            (int(start[0] * self.scale_main), int(start[1] * self.scale_main)),
            (int(end[0] * self.scale_main), int(end[1] * self.scale_main)),
            (0, 255, 0),
            2,
        )
        self.show_image(temp)

    def get_scaled_main_image(self):
        img = self.original.copy()
        return cv2.resize(img, None, fx=self.scale_main, fy=self.scale_main)

    def show_image(self, image=None):
        if image is None:
            image = self.get_scaled_main_image()
        for ellipse in self.selected_ellipse:
            cx, cy, rx, ry, angle = ellipse
            cx_s = int(cx * self.scale_main)
            cy_s = int(cy * self.scale_main)
            rx_s = int(rx * self.scale_main)
            ry_s = int(ry * self.scale_main)
            cv2.ellipse(
                image, (cx_s, cy_s), (rx_s, ry_s), angle, 0, 360, (0, 0, 255), 1
            )
        cv2.imshow(self.win_name, image)

    def process_rectangle(self, x, y, w, h):
        roi = self.original[y : y + h, x : x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced, 5)
        edges = cv2.Canny(blurred, 50, 150)
        points = np.column_stack(np.where(edges > 0))
        self.canny_points = np.flip(points, axis=1)
        self.roi_x, self.roi_y = x, y
        self.roi_w, self.roi_h = w, h

        self.current_roi_raw = roi.copy()
        self.current_roi_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        roi_disp = (
            self.current_roi_edges if self.flag == "canny" else self.current_roi_raw
        )
        self.scale_roi = min(
            self.display_size_roi[0] / roi.shape[1],
            self.display_size_roi[1] / roi.shape[0],
        )
        roi_disp = cv2.resize(roi_disp, None, fx=self.scale_roi, fy=self.scale_roi)
        self.current_roi_image = roi_disp

        cv2.namedWindow(self.roi_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.roi_win_name, *self.display_size_roi)
        cv2.moveWindow(self.roi_win_name, self.display_size_main[0] + 150, 200)
        cv2.setMouseCallback(self.roi_win_name, self.roi_mouse_handler)
        cv2.imshow(self.roi_win_name, roi_disp)

    def process_roi_points(self):
        if self.current_roi_image is None or self.canny_points is None:
            return
        filtered_points = self.selected_points
        if len(filtered_points) >= 5:
            filtered_points = np.array(filtered_points).astype(np.float32)
            ellipse = cv2.fitEllipseAMS(filtered_points)
            (cx, cy), (axes_x, axes_y), angle = ellipse
            global_cx = self.roi_x + cx
            global_cy = self.roi_y + cy
            self.selected_ellipse.append(
                (global_cx, global_cy, axes_x / 2, axes_y / 2, angle)
            )
            self.show_image()
        self.drag_start = None
        self.selected_points = []
        self.flag = "canny"

    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, *self.display_size_main)
        cv2.setMouseCallback(self.win_name, self.mouse_handler)
        self.show_image()
        while True:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == 27:  # ESC
                if self.selected_ellipse:
                    self.selected_ellipse.pop()
                    self.show_image()
                self.drag_start = None
                self.selected_points = []
            elif key == 32:  # space
                if self.drag_start is not None:
                    self.process_roi_points()
            elif key == ord("r"):
                self.flag = "raw"
                self.drag_start = None
                self.selected_points = []
        cv2.destroyAllWindows()

    def opimize_ellipse(self):
        _, selected_ellipse = optimize(self.selected_ellipse)
        self.selected_ellipse = selected_ellipse

    def save_mask(self):
        height, width = self.original.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for idx, ellipse in enumerate(self.selected_ellipse, start=1):
            cx, cy, rx, ry, angle = ellipse
            center = (int(cx), int(cy))
            axes = (int(rx), int(ry))

            # 绘制填充的椭圆，使用当前索引作为像素值
            cv2.ellipse(
                mask,
                center=center,
                axes=axes,
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=int(idx),
                thickness=-1,  # 填充
            )

        # 保存为 PNG 文件（支持 8-bit 单通道）
        cv2.imwrite(self.mask_path, mask)
        print(f"Saved mask image to {self.mask_path}")

    def save_polygon(self):
        via_data = {}
        filename = self.image_path.split("/")[-1]  # 获取当前图像文件名
        via_key = f"{filename}{self.original.size}"

        via_data[via_key] = {
            "filename": filename,
            "size": self.original.size,
            "regions": [],
            "file_attributes": {},
        }

        for ellipse in self.selected_ellipse:
            cx, cy, rx, ry, angle = ellipse
            # 生成 60 个点的多边形表示椭圆
            polygon_points = cv2.ellipse2Poly(
                center=(int(cx), int(cy)),
                axes=(int(rx), int(ry)),
                angle=int(angle),
                arcStart=0,
                arcEnd=360,
                delta=6,  # 360 / 60 = 6°，使用 delta=6 得到 60 个点
            )

            # VIA 格式要求 all_points_x 和 all_points_y
            all_points_x = [int(pt[0]) for pt in polygon_points]
            all_points_y = [int(pt[1]) for pt in polygon_points]

            via_data[via_key]["regions"].append(
                {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y,
                    },
                    "region_attributes": {},
                }
            )

        # 保存为 JSON 文件
        with open(self.polygon_path, "w") as f:
            json.dump(via_data, f, indent=2)

        print(f"Saved VIA JSON to {self.polygon_path}")

    def visualize(self):
        label_path = self.mask_path if osp.exists(self.mask_path) else self.polygon_path
        assert osp.exists(
            label_path
        ), "Label path does not exist, please annotate first!!"
        seg_visualize(self.image_path, label_path)


if __name__ == "__main__":
    img_dir = "example/"
    for img_path in glob.glob(osp.join(img_dir, "*.png")):
        detector = CircleDetector(img_path)
        detector.run()
        # detector.visualize()
        # detector.opimize_ellipse()
        # detector.save_polygon()
        # detector.save_mask()
