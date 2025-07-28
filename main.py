import glob
import threading
import numpy as np
import tkinter as tk
import os.path as osp
from tkinter import filedialog
from PIL import Image, ImageTk
from src.hole_labeling import CircleDetector


class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("圆孔标注工具")
        self.stop_flag = False

        # 获取屏幕的宽度和高度
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 设置窗口为屏幕宽度和高度的一定比例
        window_width = int(screen_width * 0.1)
        window_height = int(screen_height * 0.2)

        # 设置窗口尺寸并居中显示
        self.root.geometry(
            f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}"
        )

        self._reset_path()

        # 选择文件夹按钮
        self.select_button = tk.Button(
            root, text="选择文件", command=self.select_file_or_folder
        )
        self.select_button.pack(pady=12)

        # 开始标注按钮
        self.start_button = tk.Button(
            root, text="开始标注", command=lambda: self.start_annotation("anno")
        )
        self.start_button.pack(pady=12)

        # 结果可视化按钮
        self.show_button = tk.Button(
            root, text="结果可视化", command=lambda: self.start_annotation("show")
        )
        self.show_button.pack(pady=12)

        # 退出程序按钮
        self.end_button = tk.Button(root, text="退出程序", command=self.end_annotation)
        self.end_button.pack(pady=12)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            print(f"选择的文件夹路径: {self.folder_path}")

    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            print(f"选择的文件路径: {self.file_path}")
            
    def _reset_path(self):
        self.folder_path = ""
        self.file_path = ""

    def select_file_or_folder(self):
        # Create a dialog to choose between file or folder
        dialog = tk.Toplevel(self.root)
        dialog.title("选择类型")
        dialog.geometry("200x100")
        self._reset_path()

        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()

        file_button = tk.Button(
            dialog,
            text="选择文件",
            command=lambda: [dialog.destroy(), self.select_file()],
        )
        file_button.pack(pady=5)

        folder_button = tk.Button(
            dialog,
            text="选择文件夹",
            command=lambda: [dialog.destroy(), self.select_folder()],
        )
        folder_button.pack(pady=5)

        # Wait for dialog to close
        dialog.wait_window()

    def start_annotation(self, process_type):
        if not self.folder_path and not self.file_path:
            print("请先选择图片文件或者文件夹")
            return

        self.stop_flag = False  # 重置标志
        annotation_thread = threading.Thread(
            target=self._process,
            args=(lambda: self.stop_flag, process_type),  # 传入检查函数
        )
        annotation_thread.start()

    def _process(self, stop_flag_func, process_type="anno"):
        assert process_type in [
            "anno",
            "show",
        ], "process_type must be anno or show"
        img_paths = (
            [
                path
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
                for path in glob.glob(osp.join(self.folder_path, ext))
                if not path.endswith("_mask.png")
            ]
            if self.folder_path
            else [self.file_path]
        )

        if not img_paths:
            print("该文件夹中没有图像文件")
            return
        try:
            img_paths.sort(key=lambda x: int(osp.basename(x).split(".")[0]))
        except:
            img_paths.sort()

        has_annotated = lambda path: osp.exists(
            osp.splitext(path)[0] + "_mask.png"
        ) | osp.exists(osp.splitext(path)[0] + "_polygon.json")

        for img_path in img_paths:
            if stop_flag_func():
                print("用户请求终止标注")
                break

            if has_annotated(img_path):
                if process_type == "anno":
                    print(f"已标注: {img_path}")
                    continue
                elif process_type == "show":
                    print(f"可视化: {img_path}")
                    detector = CircleDetector(img_path)
                    vis_result = detector.visualize()
                    self._show_image_window(vis_result)
                    if stop_flag_func():
                        print("用户终止了可视化")
                        break
            else:
                if process_type == "anno":
                    print(f"正在处理: {img_path}")
                    detector = CircleDetector(img_path)
                    detector.run(stop_flag_func)
                    print(f"处理完成: {img_path}")
                elif process_type == "show":
                    print(f"请先标注：{img_path}")

    def _show_image_window(self, image_array):
        """显示图像的弹窗，支持按钮与键盘控制"""
        if image_array is None or not isinstance(image_array, np.ndarray):
            return

        # 创建弹窗窗口
        win = tk.Toplevel(self.root)
        win.title("图像可视化")
        win.geometry("1280x960")
        win.grab_set()  # 阻塞主窗口

        # 转为 ImageTk 图像
        image = Image.fromarray(image_array)
        w,h = image.size
        scale = min(1280./w, 900./h)
        image = image.resize((int(w*scale), int(h*scale)))  # 可选：缩放适配窗口
        tk_image = ImageTk.PhotoImage(image)

        # 图像显示标签
        label = tk.Label(win, image=tk_image)
        label.image = tk_image
        label.pack()

        # 按钮区
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)

        def next_image():
            win.destroy()

        def stop_all():
            self.stop_flag = True
            win.destroy()

        btn_next = tk.Button(btn_frame, text="下一张（→）", command=next_image)
        btn_next.pack(side="left", padx=20)

        btn_quit = tk.Button(btn_frame, text="退出程序（q）", command=stop_all)
        btn_quit.pack(side="left", padx=20)

        # 键盘绑定
        def on_key(event):
            if event.keysym == "Right":
                next_image()
            elif event.char == "q":
                stop_all()

        win.bind("<Key>", on_key)
        win.focus_set()  # 获取键盘焦点

        win.wait_window()  # 阻塞，直到关闭

    def end_annotation(self):
        print("退出程序")
        self.stop_flag = True
        try:
            import cv2

            cv2.destroyAllWindows()
        except:
            pass
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()
