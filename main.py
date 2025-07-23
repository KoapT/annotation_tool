import tkinter as tk
from tkinter import filedialog
from hole_labeling import CircleDetector
import os
import glob


class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("圆孔标注工具")
        # self.root.geometry("200x200")
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

    def start_annotation(self):
        if not self.folder_path:
            print("请先选择文件夹")
            return

        # 获取所有 PNG 图像路径
        img_paths = [
            path
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            for path in glob.glob(os.path.join(self.folder_path, ext))
        ]

        if not img_paths:
            print("该文件夹中没有图像文件")
            return

        for img_path in img_paths:
            print(f"正在处理: {img_path}")

            # 创建标注器并运行
            detector = CircleDetector(img_path)
            detector.run()
            detector.save_polygon()
            detector.save_mask()

            print(f"处理完成: {img_path}")

    def end_annotation(self):
        print("退出程序")
        self.root.destroy()

    def show_annotation(self):
        self.select_file()
        detector = CircleDetector(self.file_path)
        detector.visualize()



if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()
