import sys
import os
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QTabWidget,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import readGCP, readMeasure
from calculate import getCheck, getInit, getLi, getExtPre, getUnknow


class CalculationThread(QThread):
    """计算线程"""
    progress_updated = Signal(int)
    log_updated = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, gcp_file, left_file, right_file, output_dir):
        super().__init__()
        self.gcp_file = gcp_file
        self.left_file = left_file
        self.right_file = right_file
        self.output_dir = output_dir

    def run(self):
        try:
            self.log_updated.emit("正在读取数据...")
            self.progress_updated.emit(10)

            # 设置输出目录
            from calculate import set_output_dir
            set_output_dir(self.output_dir)
            self.log_updated.emit(f"输出目录: {self.output_dir}")

            mp_GCP = readGCP(self.gcp_file)
            mp_lpt = readMeasure(self.left_file)
            mp_rpt = readMeasure(self.right_file)

            self.log_updated.emit("数据读取完成")
            self.progress_updated.emit(20)

            # DLT
            self.log_updated.emit("\n===== 直接线性变换 =====")
            getCheck(mp_lpt, mp_rpt)

            self.log_updated.emit("\n--- 左片DLT ---")
            myL_l = getInit(mp_lpt, mp_GCP)
            getLi(myL_l, mp_lpt, mp_GCP, "左片")

            self.progress_updated.emit(50)

            self.log_updated.emit("\n--- 右片DLT ---")
            myL_r = getInit(mp_rpt, mp_GCP)
            getLi(myL_r, mp_rpt, mp_GCP, "右片")

            self.progress_updated.emit(70)

            # 检查点验证
            self.log_updated.emit("\n===== 检查点验证 =====")
            getExtPre(mp_lpt, mp_rpt, mp_GCP, myL_l, myL_r)

            self.progress_updated.emit(85)

            # 未知点计算
            self.log_updated.emit("\n===== 未知点计算 =====")
            getUnknow(mp_lpt, mp_rpt, myL_l, myL_r)

            self.progress_updated.emit(100)
            self.log_updated.emit("\n计算完成！")
            self.finished.emit(True, "计算成功完成")

        except Exception as e:
            self.log_updated.emit(f"\n计算错误: {str(e)}")
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("近景摄影测量处理系统 - DLT")
        self.setGeometry(100, 100, 1000, 700)

        self.init_ui()

    def init_ui(self):
        # 主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 文件选择区域
        file_layout = QHBoxLayout()

        # GCP文件
        gcp_layout = QVBoxLayout()
        self.gcp_edit = QLineEdit()
        self.gcp_edit.setPlaceholderText("控制场坐标文件")
        gcp_btn = QPushButton("浏览...")
        gcp_btn.clicked.connect(self.select_gcp_file)
        gcp_layout.addWidget(QLabel("控制场坐标文件:"))
        gcp_layout.addWidget(self.gcp_edit)
        gcp_layout.addWidget(gcp_btn)
        file_layout.addLayout(gcp_layout)

        # 左片文件
        left_layout = QVBoxLayout()
        self.left_edit = QLineEdit()
        self.left_edit.setPlaceholderText("左片像点量测文件")
        left_btn = QPushButton("浏览...")
        left_btn.clicked.connect(self.select_left_file)
        left_layout.addWidget(QLabel("左片像点量测文件:"))
        left_layout.addWidget(self.left_edit)
        left_layout.addWidget(left_btn)
        file_layout.addLayout(left_layout)

        # 右片文件
        right_layout = QVBoxLayout()
        self.right_edit = QLineEdit()
        self.right_edit.setPlaceholderText("右片像点量测文件")
        right_btn = QPushButton("浏览...")
        right_btn.clicked.connect(self.select_right_file)
        right_layout.addWidget(QLabel("右片像点量测文件:"))
        right_layout.addWidget(self.right_edit)
        right_layout.addWidget(right_btn)
        file_layout.addLayout(right_layout)

        main_layout.addLayout(file_layout)

        # 输出目录选择
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("输出目录")
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(QLabel("输出目录:"))
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_btn)
        main_layout.addLayout(output_layout)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("运行DLT计算")
        self.run_btn.clicked.connect(self.run_calculation)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        
        self.clear_btn = QPushButton("清除日志")
        self.clear_btn.clicked.connect(self.clear_log)
        
        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(self.progress_bar)

        # 结果标签页
        self.tabs = QTabWidget()
        
        # 计算日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        self.tabs.addTab(self.log_text, "计算日志")

        # DLT结果
        self.dlt_text = QTextEdit()
        self.dlt_text.setReadOnly(True)
        self.dlt_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        self.tabs.addTab(self.dlt_text, "DLT结果")

        # 待定点坐标
        self.unknown_text = QTextEdit()
        self.unknown_text.setReadOnly(True)
        self.unknown_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        self.tabs.addTab(self.unknown_text, "待定点坐标")

        main_layout.addWidget(self.tabs)

        # 设置默认文件路径
        self.set_default_paths()

    def set_default_paths(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gcp_edit.setText(os.path.join(current_dir, "data/input/控制场坐标.txt"))
        self.left_edit.setText(os.path.join(current_dir, "data/input/IMG_0015_control_points.txt"))
        self.right_edit.setText(os.path.join(current_dir, "data/input/IMG_0027_control_points.txt"))
        # 设置默认输出目录
        self.output_edit.setText(os.path.join(current_dir, "data/output"))

    def select_gcp_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择控制场坐标文件", "", "文本文件 (*.txt)")
        if file:
            self.gcp_edit.setText(file)

    def select_left_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择左片像点量测文件", "", "文本文件 (*.txt)")
        if file:
            self.left_edit.setText(file)

    def select_right_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择右片像点量测文件", "", "文本文件 (*.txt)")
        if file:
            self.right_edit.setText(file)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if dir_path:
            self.output_edit.setText(dir_path)

    def run_calculation(self):
        gcp_file = self.gcp_edit.text()
        left_file = self.left_edit.text()
        right_file = self.right_edit.text()

        if not gcp_file or not left_file or not right_file:
            QMessageBox.warning(self, "警告", "请选择所有必要的文件！")
            return

        if not os.path.exists(gcp_file):
            QMessageBox.warning(self, "警告", f"控制场坐标文件不存在: {gcp_file}")
            return

        if not os.path.exists(left_file):
            QMessageBox.warning(self, "警告", f"左片像点量测文件不存在: {left_file}")
            return

        if not os.path.exists(right_file):
            QMessageBox.warning(self, "警告", f"右片像点量测文件不存在: {right_file}")
            return

        # 获取输出目录
        output_dir = self.output_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 清空日志
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)

        # 创建并启动计算线程
        self.calc_thread = CalculationThread(gcp_file, left_file, right_file, output_dir)
        self.calc_thread.progress_updated.connect(self.update_progress)
        self.calc_thread.log_updated.connect(self.update_log)
        self.calc_thread.finished.connect(self.on_calc_finished)
        self.calc_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_calc_finished(self, success, message):
        self.run_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "完成", message)
            self.load_result_files()
        else:
            QMessageBox.error(self, "错误", f"计算失败: {message}")

    def load_result_files(self):
        # 获取输出目录
        output_dir = self.output_edit.text()
        if not output_dir:
            output_dir = "data/output"
        
        # 加载DLT结果
        left_dlt = os.path.join(output_dir, "dlt_左片.txt")
        right_dlt = os.path.join(output_dir, "dlt_右片.txt")
        
        dlt_content = ""
        if os.path.exists(left_dlt):
            with open(left_dlt, 'r', encoding='utf-8') as f:
                dlt_content += f.read() + "\n\n"
        if os.path.exists(right_dlt):
            with open(right_dlt, 'r', encoding='utf-8') as f:
                dlt_content += f.read()
        if dlt_content:
            self.dlt_text.setText(dlt_content)
        else:
            self.dlt_text.setText("DLT结果文件尚未生成")

        # 加载待定点坐标
        unknown_file = os.path.join(output_dir, "unknown_points.txt")
        if os.path.exists(unknown_file):
            with open(unknown_file, 'r', encoding='utf-8') as f:
                self.unknown_text.setText(f.read())
        else:
            self.unknown_text.setText("待定点坐标文件尚未生成")

    def clear_log(self):
        self.log_text.clear()
        self.progress_bar.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())