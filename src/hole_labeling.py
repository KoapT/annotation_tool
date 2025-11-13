import cv2
import glob
import numpy as np
import json
import os.path as osp
from .label_optimize import optimize
from .seg_visualize import seg_visualize
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from .unet import UNet
    METHOD = 'cnn'
except ImportError:
    print(
        "Warning: torch or PIL is not installed, CNN-based methods will not work."
    )
    METHOD = 'hough'


class CircleDetector:

    def __init__(self, image_path):
        self.image_path = image_path
        self.polygon_path = osp.splitext(image_path)[0] + "_polygon.json"
        self.mask_path = osp.splitext(image_path)[0] + "_mask.png"
        self.bbox_path = osp.splitext(image_path)[0] + "_bbox.txt"
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Cannot open file: {image_path}")
        self.drag_start = None
        self.selected_ellipse = []
        self.selected_ellipse_temp = []
        self.bbox = []
        self.flag = "canny"
        self.mode = "circle"
        self.roi_counter = 0
        self.read_anno()

        # 缩放相关参数
        self.display_size_main = (1280, 740)
        self.display_size_roi = (200, 200)
        self.scale_main = min(
            self.display_size_main[0] / self.original.shape[1],
            self.display_size_main[1] / self.original.shape[0],
        )

        self.win_name = f"{osp.basename(image_path)}"
        self.roi_win_name = "ROI"

        self.current_roi_image = None
        self.canny_points = None
        self.selected_points = []
        self.points2show = []

        global METHOD
        print(f"Using method: {METHOD}")
        if METHOD == 'cnn':
            self.__model_init()

    def __model_init(self):
        model_path = './src/unet/hole_ellipse.pth'
        self.model_input_size = (64, 64)
        bilinear = True
        heatmap = False
        scale = 0.5
        n_classes = 1
        self.model = UNet(n_channels=3,
                          n_classes=n_classes,
                          bilinear=bilinear,
                          scale=scale,
                          with_heatmap=heatmap)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=self.device)
        # Prefer weights_only=True (safer: avoids unpickling arbitrary objects) when supported.
        # Fall back to legacy call on older PyTorch and emit a warning.
        try:
            state_dict = torch.load(model_path,
                                    map_location=self.device,
                                    weights_only=True)
        except TypeError:
            import warnings
            warnings.warn(
                "torch.load weights_only not supported in this PyTorch version; "
                "falling back to legacy load. Ensure the model file is trusted.",
                FutureWarning,
            )
            state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def read_anno(self):
        if osp.exists(self.polygon_path):
            with open(self.polygon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            filename = osp.basename(self.image_path)
            via_key = f"{filename}"
            ellipses = data[via_key].get("ellipse", [])
            self.selected_ellipse = ellipses
            self.selected_ellipse_temp = ellipses.copy()

        if osp.exists(self.bbox_path):
            with open(self.bbox_path, 'r', encoding='utf-8') as fin:
                line = fin.readline()
                parts = line.split()
                if len(parts) == 5:
                    _, center_x, center_y, width, height = map(float, parts)
                    h, w = self.original.shape[:2]
                    x = int((center_x - width / 2) * w)
                    y = int((center_y - height / 2) * h)
                    w_box = int(width * w)
                    h_box = int(height * h)
                    self.bbox = (x, y, w_box, h_box)

    def draw_crosshair(self,
                       disp_x,
                       disp_y,
                       color=(0, 255, 0),
                       dash_len=8,
                       thickness=1):
        """
        在缩放显示图像上以 (disp_x, disp_y) 为中心绘制一水平和一垂直的虚线，
        线段覆盖整个显示图像，实时跟随鼠标移动。
        disp_x/disp_y 是窗口坐标（已经乘以 scale_main 的显示坐标）。
        """
        if not hasattr(self, "current_scaled_main_image"
                       ) or self.current_scaled_main_image is None:
            temp = self.get_scaled_main_image()
        else:
            temp = self.current_scaled_main_image.copy()
        h, w = temp.shape[:2]
        # 水平虚线
        y = int(disp_y)
        for x0 in range(0, w, dash_len * 2):
            x1 = min(x0 + dash_len, w - 1)
            cv2.line(temp, (x0, y), (x1, y), color, thickness)
        # 垂直虚线
        x = int(disp_x)
        for y0 in range(0, h, dash_len * 2):
            y1 = min(y0 + dash_len, h - 1)
            cv2.line(temp, (x, y0), (x, y1), color, thickness)
        cv2.imshow(self.win_name, temp)

    def mouse_handler(self, event, x, y, flags, param):
        # 保留原始显示坐标（窗口坐标），并还原为原图坐标
        raw_x, raw_y = int(x), int(y)
        x = int(raw_x / self.scale_main)
        y = int(raw_y / self.scale_main)

        # 鼠标移动时绘制跟随的十字虚线（使用显示坐标 raw_x/raw_y）
        if event == cv2.EVENT_MOUSEMOVE:
            self.draw_crosshair(raw_x, raw_y)
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
                    if self.mode == "circle":
                        self.process_rectangle(x, y, w, h)
                    elif self.mode == "bbox":
                        self.bbox = (x, y, w, h)
                        self.show_image()
                        self.mode = "circle"
        elif event == cv2.EVENT_MBUTTONDOWN:
            # 中键点击：如果点击位置位于某个已选椭圆内部，则删除该椭圆
            if self.selected_ellipse:
                click_x = float(x)
                click_y = float(y)
                removed = False
                for i, ellipse in enumerate(self.selected_ellipse):
                    cx, cy, rx, ry, angle = ellipse
                    # 将点变换到椭圆局部坐标系（逆旋转）
                    th = np.deg2rad(angle)
                    dx = click_x - float(cx)
                    dy = click_y - float(cy)
                    # local coords = R(-angle) * (dx,dy)
                    xl = np.cos(th) * dx + np.sin(th) * dy
                    yl = -np.sin(th) * dx + np.cos(th) * dy
                    # 判断点是否在椭圆内（rx, ry 为半轴）
                    if (xl * xl) / (float(rx) * float(rx) + 1e-12) + (
                            yl * yl) / (float(ry) * float(ry) + 1e-12) <= 1.0:
                        # 删除该椭圆并刷新显示
                        del self.selected_ellipse[i]
                        # 更新临时副本并重新显示
                        self.drag_start = None
                        self.selected_points = []
                        self.selected_ellipse_temp = self.selected_ellipse.copy(
                        )
                        self.show_image()
                        removed = True
                        break
                if not removed:
                    # 未点中任何椭圆时不做删除，但可保留其他交互（无操作）
                    pass

    def roi_mouse_handler(self, event, x, y, flags, param):
        x = int(x / self.scale_roi)
        y = int(y / self.scale_roi)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.flag == "canny":
                if self.canny_points is not None and len(
                        self.canny_points) > 0:
                    distances = np.linalg.norm(self.canny_points -
                                               np.array([x, y]),
                                               axis=1)
                    nearest_index = np.argmin(distances)
                    nearest_point = tuple(self.canny_points[nearest_index])
                    self.points2show.append(nearest_point)
            elif self.flag == "raw":
                self.points2show.append((x, y))

            if self.points2show:
                temp = self.current_roi_image.copy()
                for pt in self.points2show:
                    pt_disp = (
                        int(pt[0] * self.scale_roi),
                        int(pt[1] * self.scale_roi),
                    )
                    cv2.circle(temp, pt_disp, 3, (0, 0, 255), -1)
                cv2.imshow(self.roi_win_name, temp)

        elif event == cv2.EVENT_MBUTTONDOWN:
            if self.points2show:
                distances = np.linalg.norm(self.points2show - np.array([x, y]),
                                           axis=1)
                nearest_index = np.argmin(distances)
                del self.points2show[nearest_index]
                temp = self.current_roi_image.copy()
                for pt in self.points2show:
                    pt_disp = (
                        int(pt[0] * self.scale_roi),
                        int(pt[1] * self.scale_roi),
                    )
                    cv2.circle(temp, pt_disp, 3, (0, 0, 255), -1)
                cv2.imshow(self.roi_win_name, temp)
        if self.points2show:
            self.selected_points = self.points2show.copy()

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
        if not hasattr(self, "_scaled_image") or self._scaled_image is None:
            self._scaled_image = cv2.resize(self.original,
                                            None,
                                            fx=self.scale_main,
                                            fy=self.scale_main)
        return self._scaled_image.copy()

    def show_image(self, image=None, use_tmp=False):
        selected_ellipse = (self.selected_ellipse_temp
                            if use_tmp else self.selected_ellipse)
        if image is None:
            image = self.get_scaled_main_image()
        if self.bbox:
            x, y, w, h = self.bbox
            cv2.rectangle(
                image,
                (int(x * self.scale_main), int(y * self.scale_main)),
                (
                    int((x + w) * self.scale_main),
                    int((y + h) * self.scale_main),
                ),
                (0, 255, 0),
                2,
            )
        for i, ellipse in enumerate(selected_ellipse):
            cx, cy, rx, ry, angle = ellipse
            cx_s = int(cx * self.scale_main)
            cy_s = int(cy * self.scale_main)
            rx_s = int(rx * self.scale_main)
            ry_s = int(ry * self.scale_main)
            color = ((0, 255, 255) if i == len(selected_ellipse) - 1
                     and use_tmp == True else (0, 0, 255))
            cv2.ellipse(image, (cx_s, cy_s), (rx_s, ry_s), angle, 0, 360,
                        color, 1)
        cv2.imshow(self.win_name, image)
        if use_tmp:
            self.selected_ellipse_temp.pop()
        self.current_scaled_main_image = image

    def select_points_by_hough(self, img):
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=150,  # 高Canny阈值（突出金属圆孔）
            param2=50,  # 累加器阈值（兼顾灵敏度与精度）
            minRadius=8,
            maxRadius=120,
        )
        if circles is not None:
            circles = circles[0]
            cx, cy, r = circles[np.argmax(circles[:, -1])]
            distances = np.linalg.norm(self.canny_points - np.array([cx, cy]),
                                       axis=1)
            self.selected_points = self.canny_points[abs(distances -
                                                         r) <= (r *
                                                                0.1)].tolist()

    def find_ellipses_by_cnn(self, roi):
        raw_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        pil_img = raw_img.resize(self.model_input_size, resample=Image.BICUBIC)
        img = np.asarray(pil_img)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            mask, _ = self.model(img)
        mask = mask.cpu()
        mask = F.interpolate(mask, (raw_img.size[1], raw_img.size[0]),
                             mode="nearest")
        mask = torch.sigmoid(mask).squeeze().numpy()
        bin_mask = (mask > 0.5).astype(np.uint8) * 255  # H,W uint8
        try:
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=cv2.contourArea)
            cnt = cnt.squeeze().tolist()
        except:
            cnt = []
        self.selected_points = cnt

    def process_rectangle(self, x, y, w, h):
        global METHOD
        self.points2show = []
        roi = self.original[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced, 5)
        edges = cv2.Canny(blurred, 60, 150, L2gradient=False)
        points = np.column_stack(np.where(edges > 0))
        self.canny_points = np.flip(points, axis=1)
        self.roi_x, self.roi_y = x, y
        self.roi_w, self.roi_h = w, h

        self.current_roi_raw = roi.copy()
        self.current_roi_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.current_roi_edges[edges > 0] = (255, 0, 0)  # BGR 蓝色点
        roi_disp = (cv2.addWeighted(self.current_roi_raw, 1.0,
                                    self.current_roi_edges, 10, 0)
                    if self.flag == "canny" else self.current_roi_raw)
        self.scale_roi = min(
            self.display_size_roi[0] / roi.shape[1],
            self.display_size_roi[1] / roi.shape[0],
        )
        roi_disp = cv2.resize(roi_disp,
                              None,
                              fx=self.scale_roi,
                              fy=self.scale_roi)
        self.current_roi_image = roi_disp

        if METHOD == 'cnn':
            self.find_ellipses_by_cnn(roi)
        else:
            self.select_points_by_hough(blurred)
        self.process_roi_points(use_tmp=True)

        cv2.namedWindow(self.roi_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.roi_win_name, *self.display_size_roi)
        cv2.moveWindow(self.roi_win_name, self.display_size_main[0] + 150, 200)
        cv2.setMouseCallback(self.roi_win_name, self.roi_mouse_handler)
        cv2.imshow(self.roi_win_name, roi_disp)

    def process_roi_points(self, use_tmp=False):
        if self.current_roi_image is None or self.canny_points is None:
            return
        filtered_points = self.selected_points
        selected_ellipse = (self.selected_ellipse_temp
                            if use_tmp else self.selected_ellipse)
        if len(filtered_points) < 5:
            print("At least 5 pints are required to fit an ellipse.")
        else:
            filtered_points = np.array(filtered_points).astype(np.float32)
            ellipse = cv2.fitEllipseAMS(filtered_points)
            (cx, cy), (axes_x, axes_y), angle = ellipse
            global_cx = self.roi_x + cx
            global_cy = self.roi_y + cy
            selected_ellipse.append(
                (global_cx, global_cy, axes_x / 2, axes_y / 2, angle))
            self.show_image(use_tmp=use_tmp)

    def resort_ellipses(self):
        sorted_ellipse = sorted(self.selected_ellipse,
                                key=lambda p: (p[1], p[0]))
        sorted_ellipse.insert(0, (0, 0, 0, 0, 0))
        sorted_ellipse.append((9999, 9999, 0, 0, 0))

        # 分组：将 y 坐标接近的点分为一行
        rows = []
        current_row = [sorted_ellipse[0]]

        for i in range(1, len(sorted_ellipse)):
            if i % 3 != 0:
                current_row.append(sorted_ellipse[i])
            else:
                rows.append(current_row)
                current_row = [sorted_ellipse[i]]
        rows.append(current_row)  # 添加最后一行

        # 对每一行按 x 坐标排序（从左到右）
        for row in rows:
            row.sort(key=lambda p: p[0])

        # 按从上到下、从左到右的顺序展平
        sorted_ellipse = [p for row in rows for p in row]
        del sorted_ellipse[0]
        del sorted_ellipse[-1]
        self.selected_ellipse = sorted_ellipse

    def run(self, stop_flag_func=None):
        try:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_name, *self.display_size_main)
            cv2.setMouseCallback(self.win_name, self.mouse_handler)
            self.show_image()
            while True:
                if stop_flag_func():
                    print("用户请求终止")
                    break
                key = cv2.waitKey(1)
                if key == ord("q"):
                    self.resort_ellipses()
                    self.save_mask()
                    self.save_polygon()
                    self.save_bbox()
                    break
                elif key == 27:  # ESC
                    if self.selected_ellipse:
                        self.selected_ellipse.pop()
                        self.show_image()
                    self.drag_start = None
                    self.selected_points = []
                    self.selected_ellipse_temp = self.selected_ellipse.copy()
                elif key == 32:  # space
                    if self.drag_start is not None:
                        self.process_roi_points()
                        self.selected_points = []
                        self.selected_ellipse_temp = self.selected_ellipse.copy(
                        )
                        self.drag_start = None
                        self.flag = "canny"
                        cv2.destroyWindow(self.roi_win_name)
                elif key == ord("r"):
                    self.flag = "raw"
                    self.drag_start = None
                    self.selected_points = []
                    self.selected_ellipse_temp = self.selected_ellipse.copy()
                elif key == ord("b"):
                    self.mode = "bbox"
                    self.drag_start = None
                    self.selected_points = []
                    self.selected_ellipse_temp = self.selected_ellipse.copy()
            cv2.destroyAllWindows()
        finally:
            # 清理资源
            self._cleanup()

    def _cleanup(self):
        self.original = None
        self.current_roi_image = None
        self.current_roi_raw = None
        self.current_roi_edges = None
        self._scaled_image = None
        self.canny_points = None
        self.selected_points = []
        self.points2show = []
        self.selected_ellipse = []
        self.selected_ellipse_temp = []
        self.drag_start = None

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

    def save_bbox(self):
        """_summary_: Save with YOLO format
        """
        if self.bbox == []:
            print("Bbox is empty!!!")
            line = ''
        else:
            height, width = self.original.shape[:2]
            x, y, w, h = self.bbox
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            new_width = w / width
            new_height = h / height
            line = f"0 {center_x} {center_y} {new_width} {new_height} \n"

        with open(self.bbox_path, 'w', encoding='utf-8') as fout:
            fout.writelines(line)

    def save_polygon(self):
        height, width = self.original.shape[:2]
        via_data = {}
        filename = osp.basename(self.image_path)  # 获取当前图像文件名
        via_key = f"{filename}"

        via_data[via_key] = {
            "filename": filename,
            "image_height": height,
            "image_width": width,
            "regions": [],
            "ellipse": [],
            "file_attributes": {},
        }

        via_data[via_key]["ellipse"] = self.selected_ellipse

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

            via_data[via_key]["regions"].append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y,
                },
                "region_attributes": {},
            })

        # 保存为 JSON 文件
        with open(self.polygon_path, "w", encoding="utf-8") as f:
            json.dump(via_data, f, indent=2)

        print(f"Saved VIA JSON to {self.polygon_path}")

    def visualize(self):
        label_path = self.mask_path if osp.exists(
            self.mask_path) else self.polygon_path
        assert osp.exists(
            label_path), "Label path does not exist, please annotate first!!"
        bbox_path = self.bbox_path if osp.exists(self.bbox_path) else None
        return seg_visualize(self.image_path, label_path, bbox_path)


if __name__ == "__main__":
    img_dir = "example/"
    for img_path in glob.glob(osp.join(img_dir, "*.png")):
        detector = CircleDetector(img_path)
        detector.run()
        # detector.visualize()
        # detector.opimize_ellipse()
        # detector.save_polygon()
        # detector.save_mask()
        # detector.save_bbox()
