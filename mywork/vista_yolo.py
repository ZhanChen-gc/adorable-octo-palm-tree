import sys
import json
import csv
import os
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt, QThread, Signal, Slot, QTimer, QSize
)
from PySide6.QtGui import (
    QAction, QPixmap, QImage, QFont, QColor, QBrush, QPen, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QFileDialog, QComboBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
    QButtonGroup, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QMessageBox, QSplitter, QTabWidget,
    QLineEdit, QFormLayout, QFrame
)
from ultralytics import YOLO


# ==================== 推理线程基类 ====================
class InferenceThread(QThread):
    """处理所有推理任务（图片/视频/摄像头/目录）的线程基类"""
    frame_processed = Signal(np.ndarray, dict)  # 处理后的图像（BGR），结构化数据
    finished = Signal()
    error = Signal(str)

    def __init__(self, model, source, conf_thres, iou_thres):
        super().__init__()
        self.model = model
        self.source = source
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        """由子类重写具体推理逻辑"""
        raise NotImplementedError


class ImageInferenceThread(InferenceThread):
    """单张图片推理线程"""
    def run(self):
        try:
            results = self.model(
                self.source,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False
            )
            if not results:
                self.error.emit("模型推理失败，未返回结果")
                return

            result = results[0]
            # 获取带检测框的图像（BGR格式）
            img_bgr = result.plot()  # 返回BGR numpy数组
            # 提取结构化数据
            data = self._extract_data(result)
            self.frame_processed.emit(img_bgr, data)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _extract_data(self, result):
        """从YOLO结果中提取结构化数据"""
        boxes_data = []
        if result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            names = result.names
            for i in range(len(boxes)):
                box = xyxy[i].tolist()
                class_id = int(cls[i])
                class_name = names[class_id]
                confidence = float(conf[i])
                boxes_data.append({
                    "bbox": box,        # [x1, y1, x2, y2]
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence
                })
        return {"boxes": boxes_data, "frame_index": 0}


class VideoInferenceThread(InferenceThread):
    """视频/摄像头推理线程（连续帧）"""
    def __init__(self, model, source, conf_thres, iou_thres, is_camera=False):
        super().__init__(model, source, conf_thres, iou_thres)
        self.is_camera = is_camera
        self.cap = None
        self.fps = 30
        self.frame_count = 0

    def run(self):
        try:
            if self.is_camera:
                # 摄像头：source可以是整数（设备ID）或字符串（RTSP地址）
                if isinstance(self.source, int) or self.source.isdigit():
                    cap_index = int(self.source) if isinstance(self.source, str) else self.source
                    self.cap = cv2.VideoCapture(cap_index)
                else:
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(str(self.source))
            if not self.cap.isOpened():
                self.error.emit("无法打开视频源")
                return

            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_count += 1
                # 推理
                results = self.model(
                    frame,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    verbose=False
                )
                if results:
                    result = results[0]
                    img_bgr = result.plot()
                    data = self._extract_data(result, self.frame_count)
                    self.frame_processed.emit(img_bgr, data)
                else:
                    # 无检测结果，直接显示原图
                    self.frame_processed.emit(frame, {"boxes": [], "frame_index": self.frame_count})

                # 控制帧率（可选）
                # time.sleep(1/self.fps)

            self.cap.release()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _extract_data(self, result, frame_idx):
        boxes_data = []
        if result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            names = result.names
            for i in range(len(boxes)):
                box = xyxy[i].tolist()
                class_id = int(cls[i])
                class_name = names[class_id]
                confidence = float(conf[i])
                boxes_data.append({
                    "bbox": box,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence
                })
        return {"boxes": boxes_data, "frame_index": frame_idx}


class BatchImageInferenceThread(InferenceThread):
    """批量目录推理线程（依次处理每个图片，通过信号发送）"""
    def __init__(self, model, source_dir, conf_thres, iou_thres):
        super().__init__(model, source_dir, conf_thres, iou_thres)
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def run(self):
        try:
            source_path = Path(self.source)
            if not source_path.is_dir():
                self.error.emit("批量目录路径无效")
                return
            # 收集所有图片文件
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(source_path.glob(f"*{ext}"))
                image_files.extend(source_path.glob(f"*{ext.upper()}"))
            if not image_files:
                self.error.emit("目录中未找到图片文件")
                return
            for img_path in image_files:
                if not self._running:
                    break
                try:
                    results = self.model(
                        str(img_path),
                        conf=self.conf_thres,
                        iou=self.iou_thres,
                        verbose=False
                    )
                    if results:
                        result = results[0]
                        img_bgr = result.plot()
                        data = self._extract_data(result, str(img_path))
                        self.frame_processed.emit(img_bgr, data)
                    else:
                        # 无检测，但还需显示原图？
                        img = cv2.imread(str(img_path))
                        self.frame_processed.emit(img, {"boxes": [], "file_path": str(img_path)})
                except Exception as e:
                    self.error.emit(f"处理图片 {img_path.name} 时出错: {e}")
                # 控制速度，避免GUI过载
                time.sleep(0.05)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _extract_data(self, result, file_path):
        boxes_data = []
        if result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            names = result.names
            for i in range(len(boxes)):
                box = xyxy[i].tolist()
                class_id = int(cls[i])
                class_name = names[class_id]
                confidence = float(conf[i])
                boxes_data.append({
                    "bbox": box,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence
                })
        return {"boxes": boxes_data, "file_path": file_path}


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO智能目标检测可视化系统")
        self.setMinimumSize(1200, 800)

        self.model = None          # YOLO模型实例
        self.current_source = None # 当前数据源（文件路径/目录/摄像头标识）
        self.current_inference_thread = None
        self.current_frame = None   # 当前显示的图像（BGR）
        self.current_boxes_data = None # 当前帧的结构化数据

        self.init_ui()

    def init_ui(self):
        # 中央widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 左控制面板
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        # 模型管理组
        model_group = QGroupBox("模型管理")
        model_layout = QVBoxLayout()
        self.model_path_label = QLabel("未加载模型")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: gray;")
        load_btn = QPushButton("加载模型 (.pt)")
        load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_btn)
        model_layout.addWidget(self.model_path_label)
        model_group.setLayout(model_layout)

        # 数据源组
        source_group = QGroupBox("数据源")
        source_layout = QVBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["单张图片", "视频文件", "批量目录", "摄像头"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        self.source_selector = QWidget()
        self.source_selector_layout = QHBoxLayout(self.source_selector)
        self.source_selector_layout.setContentsMargins(0, 0, 0, 0)
        self.source_file_edit = QLineEdit()
        self.source_file_edit.setPlaceholderText("请选择...")
        self.source_browse_btn = QPushButton("浏览")
        self.source_browse_btn.clicked.connect(self.browse_source)
        self.source_selector_layout.addWidget(self.source_file_edit)
        self.source_selector_layout.addWidget(self.source_browse_btn)
        source_layout.addWidget(self.source_combo)
        source_layout.addWidget(self.source_selector)
        # 摄像头额外配置（可选）
        self.camera_config = QWidget()
        cam_layout = QHBoxLayout(self.camera_config)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.addWidget(QLabel("设备ID/URL:"))
        self.camera_id_edit = QLineEdit()
        self.camera_id_edit.setPlaceholderText("0 或 rtsp://...")
        cam_layout.addWidget(self.camera_id_edit)
        self.camera_config.setVisible(False)
        source_layout.addWidget(self.camera_config)
        source_group.setLayout(source_layout)

        # 检测参数组
        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_spin.setValue(v/100.0))
        self.conf_spin.valueChanged.connect(lambda v: self.conf_slider.setValue(int(v*100)))
        param_layout.addWidget(self.conf_slider, 0, 1)
        param_layout.addWidget(self.conf_spin, 0, 2)

        param_layout.addWidget(QLabel("IoU阈值:"), 1, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(0.45)
        self.iou_slider.valueChanged.connect(lambda v: self.iou_spin.setValue(v/100.0))
        self.iou_spin.valueChanged.connect(lambda v: self.iou_slider.setValue(int(v*100)))
        param_layout.addWidget(self.iou_slider, 1, 1)
        param_layout.addWidget(self.iou_spin, 1, 2)
        param_group.setLayout(param_layout)

        # 控制按钮
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_inference)

        # 进度条（用于批量处理提示）
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        left_layout.addWidget(model_group)
        left_layout.addWidget(source_group)
        left_layout.addWidget(param_group)
        left_layout.addWidget(self.start_btn)
        left_layout.addWidget(self.stop_btn)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()

        # 中：显示区域
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        self.display_label = QLabel("检测画面将显示于此")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b; color: white;")
        self.display_label.setScaledContents(False)
        middle_layout.addWidget(self.display_label)
        # 视频控制栏（播放/暂停，仅用于视频/摄像头）
        self.video_controls = QHBoxLayout()
        self.play_pause_btn = QPushButton("暂停")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.video_controls.addStretch()
        self.video_controls.addWidget(self.play_pause_btn)
        middle_layout.addLayout(self.video_controls)

        # 右侧：结果展示与导出
        right_panel = QWidget()
        right_panel.setMaximumWidth(450)
        right_layout = QVBoxLayout(right_panel)

        # 表格显示结构化数据
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "x1", "y1", "x2", "y2"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(QLabel("检测结果列表"))
        right_layout.addWidget(self.table)

        # 导出按钮
        export_group = QGroupBox("导出结果")
        export_layout = QHBoxLayout()
        self.save_img_btn = QPushButton("保存检测图像")
        self.save_img_btn.clicked.connect(self.save_image)
        self.save_json_btn = QPushButton("导出JSON")
        self.save_json_btn.clicked.connect(self.save_json)
        self.save_csv_btn = QPushButton("导出CSV")
        self.save_csv_btn.clicked.connect(self.save_csv)
        export_layout.addWidget(self.save_img_btn)
        export_layout.addWidget(self.save_json_btn)
        export_layout.addWidget(self.save_csv_btn)
        export_group.setLayout(export_layout)
        right_layout.addWidget(export_group)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)

        # 组装主布局
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(middle)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([350, 700, 400])
        main_layout.addWidget(main_splitter)

    # ---------- 模型加载 ----------
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLOv8权重文件", "", "PyTorch模型 (*.pt)"
        )
        if not file_path:
            return
        try:
            self.status_label.setText("正在加载模型...")
            self.model = YOLO(file_path)
            self.model_path_label.setText(f"已加载: {os.path.basename(file_path)}")
            self.model_path_label.setStyleSheet("color: green;")
            self.status_label.setText("模型加载成功")
            QMessageBox.information(self, "成功", f"模型 {os.path.basename(file_path)} 加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\\n{e}")
            self.status_label.setText("模型加载失败")

    # ---------- 数据源交互 ----------
    def on_source_changed(self, text):
        if text == "摄像头":
            self.source_selector.setVisible(False)
            self.camera_config.setVisible(True)
        else:
            self.source_selector.setVisible(True)
            self.camera_config.setVisible(False)
            # 清空旧路径
            self.source_file_edit.clear()

    def browse_source(self):
        source_type = self.source_combo.currentText()
        if source_type == "单张图片":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
            )
        elif source_type == "视频文件":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
        elif source_type == "批量目录":
            path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        else:
            path = ""
        if path:
            self.source_file_edit.setText(path)

    # ---------- 开始/停止推理 ----------
    def start_inference(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        # 获取数据源
        source_type = self.source_combo.currentText()
        if source_type == "摄像头":
            camera_input = self.camera_id_edit.text().strip()
            if not camera_input:
                camera_input = "0"  # 默认第一个摄像头
            self.current_source = camera_input
        else:
            file_path = self.source_file_edit.text().strip()
            if not file_path:
                QMessageBox.warning(self, "警告", "请选择数据源")
                return
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "警告", "数据源不存在")
                return
            self.current_source = file_path

        # 获取参数
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()

        # 停止已有线程
        self.stop_inference()

        # 根据类型创建线程
        if source_type == "单张图片":
            thread = ImageInferenceThread(self.model, self.current_source, conf, iou)
        elif source_type == "视频文件":
            thread = VideoInferenceThread(self.model, self.current_source, conf, iou, is_camera=False)
        elif source_type == "批量目录":
            thread = BatchImageInferenceThread(self.model, self.current_source, conf, iou)
        elif source_type == "摄像头":
            thread = VideoInferenceThread(self.model, self.current_source, conf, iou, is_camera=True)
        else:
            return

        thread.frame_processed.connect(self.update_display)
        thread.finished.connect(self.on_inference_finished)
        thread.error.connect(self.on_inference_error)

        self.current_inference_thread = thread
        thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.play_pause_btn.setEnabled(source_type in ["视频文件", "摄像头"])  # 启用暂停按钮
        self.status_label.setText("推理进行中...")

    def stop_inference(self):
        if self.current_inference_thread is not None and self.current_inference_thread.isRunning():
            self.current_inference_thread.stop()
            self.current_inference_thread.wait(1000)  # 等待1秒
            self.current_inference_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.play_pause_btn.setEnabled(False)
        self.status_label.setText("已停止")

    def on_inference_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.play_pause_btn.setEnabled(False)
        self.status_label.setText("推理结束")
        self.current_inference_thread = None

    def on_inference_error(self, err_msg):
        QMessageBox.critical(self, "推理错误", err_msg)
        self.on_inference_finished()

    # ---------- 显示更新 ----------
    @Slot(np.ndarray, dict)
    def update_display(self, img_bgr, data):
        """接收推理线程传来的图像和结构化数据，更新界面"""
        if img_bgr is None:
            return
        self.current_frame = img_bgr
        self.current_boxes_data = data.get("boxes", [])

        # 更新表格
        self.update_table(data.get("boxes", []))

        # 显示图像（BGR -> RGB -> QImage -> QPixmap）
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        # 缩放以适合显示区域
        scaled_pixmap = pixmap.scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled_pixmap)

    def update_table(self, boxes_data):
        self.table.setRowCount(len(boxes_data))
        for row, box in enumerate(boxes_data):
            self.table.setItem(row, 0, QTableWidgetItem(box["class_name"]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{box['confidence']:.3f}"))
            # 坐标四舍五入保留整数
            x1, y1, x2, y2 = map(int, box["bbox"])
            self.table.setItem(row, 2, QTableWidgetItem(str(x1)))
            self.table.setItem(row, 3, QTableWidgetItem(str(y1)))
            self.table.setItem(row, 4, QTableWidgetItem(str(x2)))
            self.table.setItem(row, 5, QTableWidgetItem(str(y2)))
        self.table.resizeColumnsToContents()

    # ---------- 导出功能 ----------
    def save_image(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        default_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path, _ = QFileDialog.getSaveFileName(self, "保存检测图像", default_name, "JPEG图像 (*.jpg);;PNG图像 (*.png)")
        if path:
            cv2.imwrite(path, self.current_frame)
            QMessageBox.information(self, "成功", f"图像已保存至 {path}")

    def save_json(self):
        if self.current_boxes_data is None:
            QMessageBox.warning(self, "警告", "没有检测数据可导出")
            return
        default_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(self, "导出JSON", default_name, "JSON文件 (*.json)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.current_boxes_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", f"数据已导出至 {path}")

    def save_csv(self):
        if self.current_boxes_data is None:
            QMessageBox.warning(self, "警告", "没有检测数据可导出")
            return
        default_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", default_name, "CSV文件 (*.csv)")
        if path:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["class_name", "confidence", "x1", "y1", "x2", "y2"])
                for box in self.current_boxes_data:
                    x1, y1, x2, y2 = box["bbox"]
                    writer.writerow([box["class_name"], box["confidence"], x1, y1, x2, y2])
            QMessageBox.information(self, "成功", f"数据已导出至 {path}")

    def toggle_play_pause(self):
        """暂停/继续视频/摄像头流"""
        if self.current_inference_thread is None:
            return
        if hasattr(self.current_inference_thread, '_running'):
            if self.current_inference_thread._running:
                self.current_inference_thread.stop()
                self.play_pause_btn.setText("继续")
                self.status_label.setText("已暂停")
            else:
                # 重启线程？简单起见，重新开始新线程
                self.start_inference()
        # 注意：实际应用中需要更复杂的控制，这里仅作示意


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())