import glob
import threading
import tkinter as tk
import os.path as osp
from tkinter import filedialog
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

        self.folder_path = ""
        self.file_path = ""

        # 选择文件夹按钮
        self.select_button = tk.Button(
            root, text="选择文件夹", command=self.select_folder
        )
        self.select_button.pack(pady=12)

        # 开始标注按钮
        self.start_button = tk.Button(
            root, text="开始标注", command=self.start_annotation
        )
        self.start_button.pack(pady=12)

        # 退出程序按钮
        self.end_button = tk.Button(root, text="退出程序", command=self.end_annotation)
        self.end_button.pack(pady=12)

        # 结果可视化按钮
        self.show_button = tk.Button(
            root, text="结果可视化", command=self.show_annotation
        )
        self.show_button.pack(pady=12)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            print(f"选择的文件夹路径: {self.folder_path}")

    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            print(f"选择的文件路径: {self.file_path}")

    def select_file_or_folder(self):
        # Create a dialog to choose between file or folder
        dialog = tk.Toplevel(self.root)
        dialog.title("选择类型")
        dialog.geometry("200x100")

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

    def start_annotation(self):
        if not self.folder_path:
            print("请先选择文件夹")
            return

        self.stop_flag = False  # 重置标志
        annotation_thread = threading.Thread(
            target=self._process_annotations,
            args=(lambda: self.stop_flag,),  # 传入检查函数
        )
        annotation_thread.start()

    def _process_annotations(self, stop_flag_func):
        img_paths = [
            path
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            for path in glob.glob(osp.join(self.folder_path, ext))
            if not (
                path.endswith("_mask.png")
                | osp.exists(osp.splitext(path)[0] + "_mask.png")
                | osp.exists(osp.splitext(path)[0] + "_polygon.json")
            )
        ]
        try:
            img_paths.sort(key=lambda x: int(osp.basename(x).split(".")[0]))
        except:
            img_paths.sort()

        if not img_paths:
            print("该文件夹中没有图像文件")
            return

        for img_path in img_paths:
            if stop_flag_func():
                print("用户请求终止标注")
                break

            print(f"正在处理: {img_path}")

            detector = CircleDetector(img_path)
            detector.run()
            detector.save_polygon()
            detector.save_mask()

            print(f"处理完成: {img_path}")

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

    def show_annotation(self):
        # Ask user to choose file or folder
        choice = tk.messagebox.askquestion(
            "选择类型", "是否选择文件夹？\n选择'是'选择文件夹，'否'选择文件"
        )

        if choice == "yes":
            # Select folder and visualize all images
            self.select_folder()
            if not self.folder_path:
                return

            # Get all image paths in folder
            img_paths = [
                path
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
                for path in glob.glob(osp.join(self.folder_path, ext))
                if osp.exists(osp.splitext(path)[0] + "_mask.png")
                | osp.exists(osp.splitext(path)[0] + "_polygon.json")
            ]

            # Sort images
            try:
                img_paths.sort(key=lambda x: int(osp.basename(x).split(".")[0]))
            except:
                img_paths.sort()

            # Visualize each image
            for img_path in img_paths:
                print(f"可视化: {img_path}")
                detector = CircleDetector(img_path)
                detector.visualize()
        else:
            # Select file and visualize
            self.select_file()
            if self.file_path:
                print(f"可视化: {self.file_path}")
                detector = CircleDetector(self.file_path)
                detector.visualize()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()
