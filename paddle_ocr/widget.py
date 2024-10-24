import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from urllib.request import urlopen
import collections
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QGraphicsScene, QGraphicsPixmapItem

from utils import pip_plugins
from widgets.plugins_install_dialog import PlugInstallDialog
# import cv2
from widgets.qfluentwidgets import PushButton, qconfig
import requests
# import numpy as np
import tarfile
from widgets.setting_interface import SystemConfig


class Widget(QObject):

    def setupUi(self, MainWindow):
        # MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(800, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = MainWindow
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_3.addWidget(self.label_7)
        self.targetCombox = QtWidgets.QComboBox(self.centralwidget)
        self.targetCombox.setObjectName("targetCombox")
        self.targetCombox.addItem("")
        self.targetCombox.addItem("")
        self.horizontalLayout_3.addWidget(self.targetCombox)
        self.mainBtn = QtWidgets.QPushButton(self.centralwidget)
        self.mainBtn.setObjectName("mainBtn")
        self.horizontalLayout_3.addWidget(self.mainBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.downProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.downProgressBar.setEnabled(True)
        self.downProgressBar.setProperty("value", 0)
        self.downProgressBar.setObjectName("downProgressBar")
        self.verticalLayout_2.addWidget(self.downProgressBar)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.resultImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultImageLabel.setText("")
        self.resultImageLabel.setObjectName("resultImageLabel")
        self.horizontalLayout_4.addWidget(self.resultImageLabel)
        self.resultLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel.setText("")
        self.resultLabel.setObjectName("resultLabel")
        self.horizontalLayout_4.addWidget(self.resultLabel)
        self.horizontalLayout_4.setStretch(0, 6)
        self.horizontalLayout_4.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.setStretch(2, 1)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(MainWindow)
        self.mainBtn.clicked.connect(lambda: self.main_btn_clicked())  # type: ignore

        self.targetCombox.currentIndexChanged['int'].connect(
            lambda: self.combox_index_changed(self.targetCombox))  # type: ignore

        self.window = MainWindow
        self.init_widget()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_7.setText(_translate("MainWindow", "Target"))
        self.targetCombox.setItemText(0, _translate("MainWindow", "local image"))
        self.targetCombox.setItemText(1, _translate("MainWindow", "find camera"))
        self.mainBtn.setText(_translate("MainWindow", "Start"))

    def init_widget(self):
        self.update_status_signal.connect(self.update_status_callback)
        self.downProgressBar.setVisible(False)
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    all_model_download_url = {
        "ch_PP-OCRv3_det_infer": "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/paddle-ocr/ch_PP-OCRv3_det_infer.tar",
        "ch_PP-OCRv3_rec_infer": "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/paddle-ocr/ch_PP-OCRv3_rec_infer.tar"
    }

    find_camera_ing = False
    def combox_index_changed(self,source):
        print(f"combox_index_changed:{source.objectName()}")
        if source.objectName() == "targetCombox":
            if self.find_camera_ing:
                return
            if self.targetCombox.currentIndex() == self.targetCombox.count() - 1:
                print("find new camera list")
                self.find_camera_ing = True
                self.targetCombox.clear()
                self.targetCombox.addItem("local image")
                count = self.get_camera_count()
                print(f"camera count:{count - 1}")
                for i in range(count - 1):
                    self.targetCombox.addItem(f"Camera:{i}")
                # self.targetCombox.addItem("相机0")
                self.targetCombox.addItem("find camera")
                self.find_camera_ing = False

    def get_camera_count(self):
        # 获取可用的摄像头数量
        import cv2
        count = 0
        while True:
            # 打开摄像头
            cap = cv2.VideoCapture(count)
            if not cap.isOpened():
                # 获取摄像头的名称
                return count
            count+=1
            # 释放摄像头资源
            cap.release()

    def main_btn_clicked(self):
        print("main btn click")
        self.not_install_pkgs = []
        if not pip_plugins.verify_library_installed("cv2"):
            print("not install package:opencv")
            self.not_install_pkgs.append("opencv-python")
        if not pip_plugins.verify_library_installed("paddle"):
            print("not install package:paddle")
            self.not_install_pkgs.append("paddlepaddle")
        if not pip_plugins.verify_library_installed("shapely"):
            print("not install package:shapely")
            self.not_install_pkgs.append("shapely>=1.7.1")
        if not pip_plugins.verify_library_installed("pyclipper"):
            print("not install package:pyclipper")
            self.not_install_pkgs.append("pyclipper>=1.2.1")
        if len(self.not_install_pkgs) > 0:
            self.install_dialog = PlugInstallDialog()
            self.install_dialog.show()
            for pkg in self.not_install_pkgs:
                self.install_dialog.addProgressBar(pkg)
            print(f"start thread:{self.not_install_pkgs}")
            t = threading.Thread(target=self.show_install_lib_dialog)
            t.start()
            return
        print("all library installed")
        if self.mainBtn.text() == "Stop":
            self.running = False
            self.update_status_signal.emit(8, 0, "", 0)
            return
        det_model_file_path = Path("model/ch_PP-OCRv3_det_infer/inference.pdmodel")
        rec_model_file_path = Path("model/ch_PP-OCRv3_rec_infer/inference.pdmodel")
        if not det_model_file_path.exists() or not rec_model_file_path.exists():
            print(f"local not exists OCRv3_det_infer:{det_model_file_path.exists()}.not exists OCRv3_rec_infer:{rec_model_file_path.exists()}.start download")
            # det_model.export(format="openvino", dynamic=True, half=True)
            self.downProgressBar.setValue(0)
            self.downProgressBar.setVisible(True)
            threading.Thread(target=self.download_model_thread).start()
            self.mainBtn.setEnabled(False)
            return
        target_device = self.targetCombox.currentText()
        self.system_config = SystemConfig()
        qconfig.load("config.json", self.system_config)
        if target_device == "local image":
            fname, _tmp = QFileDialog.getOpenFileNames(self.window, 'Open file', './images', "*.png *.ico *.jpg")
            print(f"select image:{fname}")
            if len(fname) < 1:
                print("not selected file")
                return
            import cv2
            import openvino as ov
            import openvino_desktop_widgets.paddle_ocr.pre_post_processing as processing
            import copy
            device = self.system_config.device
            # Initialize OpenVINO Runtime for text detection.
            core = ov.Core()
            det_model = core.read_model(model=det_model_file_path)
            det_compiled_model = core.compile_model(model=det_model, device_name=device.value)

            # Get input and output nodes for text detection.
            det_input_layer = det_compiled_model.input(0)
            det_output_layer = det_compiled_model.output(0)
            # Read the model and corresponding weights from a file.
            rec_model = core.read_model(model=rec_model_file_path)

            # Assign dynamic shapes to every input layer on the last dimension.
            for input_layer in rec_model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[3] = -1
                rec_model.reshape({input_layer: input_shape})

            rec_compiled_model = core.compile_model(model=rec_model, device_name=device.value)
            #
            # Get input and output nodes.
            rec_input_layer = rec_compiled_model.input(0)
            rec_output_layer = rec_compiled_model.output(0)
            frame = cv2.imread(fname[0])

            processing_times = collections.deque()
            # If the frame is larger than full HD, reduce size to improve the performance.
            draw_img = processing.PaddleOCR().process_frame(frame, processing_times, det_compiled_model, det_output_layer, rec_compiled_model, rec_output_layer)
            img = cv2.resize(draw_img,
                             (self.resultImageLabel.width(), self.resultImageLabel.height()))
            self.resultImageLabel.setPixmap(QPixmap.fromImage(self.mat_to_image(img)))
        elif target_device != "find camera":
            self.camera_index = self.targetCombox.currentIndex() - 1
            self.running = True
            threading.Thread(target=self.camera_work).start()
            self.mainBtn.setText("Stop")

    running = False
    def camera_work(self):
        import cv2
        import openvino as ov
        import openvino_desktop_widgets.paddle_ocr.pre_post_processing as processing
        import copy
        det_model_file_path = Path("model/ch_PP-OCRv3_det_infer/inference.pdmodel")
        rec_model_file_path = Path("model/ch_PP-OCRv3_rec_infer/inference.pdmodel")
        if not det_model_file_path.exists() or not rec_model_file_path.exists():
            print(
                f"local not exists OCRv3_det_infer:{det_model_file_path.exists()}.not exists OCRv3_rec_infer:{rec_model_file_path.exists()}.start download")
            # det_model.export(format="openvino", dynamic=True, half=True)
            self.downProgressBar.setValue(0)
            self.downProgressBar.setVisible(True)
            threading.Thread(target=self.download_model_thread).start()
            self.mainBtn.setEnabled(False)
            return
        self.system_config = SystemConfig()
        qconfig.load("config.json", self.system_config)
        device = self.system_config.device
        # Initialize OpenVINO Runtime for text detection.
        core = ov.Core()
        det_model = core.read_model(model=det_model_file_path)
        det_compiled_model = core.compile_model(model=det_model, device_name=device.value)

        # Get input and output nodes for text detection.
        det_input_layer = det_compiled_model.input(0)
        det_output_layer = det_compiled_model.output(0)
        # Read the model and corresponding weights from a file.
        rec_model = core.read_model(model=rec_model_file_path)

        # Assign dynamic shapes to every input layer on the last dimension.
        for input_layer in rec_model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[3] = -1
            rec_model.reshape({input_layer: input_shape})

        rec_compiled_model = core.compile_model(model=rec_model, device_name=device.value)
        #
        # Get input and output nodes.
        rec_input_layer = rec_compiled_model.input(0)
        rec_output_layer = rec_compiled_model.output(0)

        cap = cv2.VideoCapture(int(self.camera_index))
        try:
            # Create a video player to play with target fps.
            # Start capturing.
            import numpy as np
            processing_times = collections.deque()
            while self.running:
                # Grab the frame.
                hx,frame = cap.read()
                if frame is None:
                    print("Source ended")
                    break
                # If the frame is larger than full HD, reduce size to improve the performance.
                draw_img = processing.PaddleOCR().process_frame(frame, processing_times, det_compiled_model,
                                                                det_output_layer,
                                                                rec_compiled_model, rec_output_layer)

                img = cv2.resize(draw_img,
                                 (self.resultImageLabel.width(), self.resultImageLabel.height()))
                self.resultImageLabel.setPixmap(QPixmap.fromImage(self.mat_to_image(img)))
        except KeyboardInterrupt:
            print("Interrupted")
            # any different error
        except RuntimeError as e:
            print(e)
        finally:
            cap.release()

    update_status_signal = pyqtSignal(int, int, str, int)

    def mat_to_image(self, mat):
        if len(mat.shape) == 2:
            rows,cols = mat.shape
            bytesPerLine = cols
            return QImage(mat.data,cols,rows,bytesPerLine,QImage.Format_Indexed8)
        else:
            rows, cols,channels = mat.shape
            bytesPerLine = channels * cols
            return QImage(mat.data, cols, rows, bytesPerLine, QImage.Format_RGB888)

    def update_status_callback(self, type, index, text, process):
        if type == 1:
            self.install_dialog.addProgressBar(text)
        elif type == 2:
            self.install_dialog.addProgressBarArray[index]["label"].setText(text)
            self.install_dialog.addProgressBarArray[index]["bar"].setValue(process)
        elif type == 4:
            print(f"close dialog.{self.install_dialog is not None}")
            if  self.install_dialog is not None:
                self.install_dialog.reject()
            self.install_dialog = None
        elif type == 5:
            reply = QMessageBox.question(self.window, 'Tip',
                                         "There is no model file locally, or the model file download failed.\nYou can try again or use scientific Internet access.\nIt may also be that there is not enough memory.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            # 调用一个简单的shell命令，比如ls
            subprocess.run(["taskkill", "/f", "/im", "LLM.exe"], shell=True)
        elif type == 6:
            if not self.install_dialog:
                self.install_dialog.reject()
            self.install_dialog = None
            reply = QMessageBox.question(self.window, 'Tip',
                                         "Failed to download the dependent library.\nYou can try again or use scientific Internet access.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            # 调用一个简单的shell命令，比如ls
            subprocess.run(["taskkill", "/f", "/im", "LLM.exe"], shell=True)
        elif type == 7:
            self.downProgressBar.setValue(index)
        elif type == 8:
            self.downProgressBar.setVisible(False)
            self.mainBtn.setEnabled(True)
            self.mainBtn.setText("Start")
        elif type == 9:
            QMessageBox.question(self.window, 'Tip',text,
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

    def run_model_download(self,dst: str, model_file_path: Path) -> None:
        """
        Download pre-trained models from PaddleOCR resources

        Parameters:
            model_url: url link to pre-trained models
            model_file_path: file path to store the downloaded model
        """
        archive_path = dst
        file = tarfile.open(archive_path)
        res = file.extractall(model_file_path.parent)
        file.close()
        if not res:
            print(f"Model Extracted to {model_file_path}.")
        else:
            print("Error Extracting the model. Please check the network.")

    def download_model_thread(self):
        # 获取文件长度
        det_model_file_path = Path("model/ch_PP-OCRv3_det_infer/inference.pdmodel")
        if not det_model_file_path.exists():
            start = time.time()  # 下载开始时间
            url = self.all_model_download_url["ch_PP-OCRv3_det_infer"]
            dst = f"ch_PP-OCRv3_det_infer.tar"
            print(f"start download model OCRv3_det_infer.{url}")
            try:
                response = requests.get(url, stream=True)  # stream=True必须写上
                size = 0  # 初始化已下载大小
                chunk_size = 1024  # 每次下载的数据大小
                content_size = int(response.headers['content-length'])  # 下载文件总大小
                if response.status_code == 200:  # 判断是否响应成功
                    print('Start downloading, [file size]: {size:.2f}MB'.format(
                        size=content_size / chunk_size / 1024))  # 开始下载，显示下载文件大小
                    # filepath = '下载/222.mp4'  #注：必须加上扩展名
                    with open(dst, 'wb') as file:  # 显示进度条
                        for data in response.iter_content(chunk_size=chunk_size):
                            file.write(data)
                            size += len(data)
                            self.update_status_signal.emit(7, int(size / content_size * 100), "",
                                                           0)
                            # print('\r' + '[下载进度]:%s%.2f%%' % (
                            #     '>' * int(size * 50 / content_size), float(size / content_size * 100)), end=' ')
                end = time.time()  # 下载结束时间
                print('Done! Duration:%.2f seconds' % (end - start))  # 输出下载用时时间
                if os.path.exists(dst):
                    print(f"start export model to path")
                    self.run_model_download(dst,Path("model/ch_PP-OCRv3_det_infer"))
                    os.remove(dst)
            except Exception as e:
                print(traceback.print_exc())
                print("error, accessing url:%s exception" % url)
                self.update_status_signal.emit(8, 0, "", 0)
            rec_model_file_path = Path("model/ch_PP-OCRv3_rec_infer/inference.pdmodel")
            if not rec_model_file_path.exists():
                start = time.time()  # 下载开始时间
                url = self.all_model_download_url["ch_PP-OCRv3_rec_infer"]
                dst = f"ch_PP-OCRv3_rec_infer.tar"
                print(f"start download model ch_PP-OCRv3_rec_infer.{url}")
                try:
                    response = requests.get(url, stream=True)  # stream=True必须写上
                    size = 0  # 初始化已下载大小
                    chunk_size = 1024  # 每次下载的数据大小
                    content_size = int(response.headers['content-length'])  # 下载文件总大小
                    if response.status_code == 200:  # 判断是否响应成功
                        print('Start downloading, [file size]: {size:.2f}MB'.format(
                            size=content_size / chunk_size / 1024))  # 开始下载，显示下载文件大小
                        # filepath = '下载/222.mp4'  #注：必须加上扩展名
                        with open(dst, 'wb') as file:  # 显示进度条
                            for data in response.iter_content(chunk_size=chunk_size):
                                file.write(data)
                                size += len(data)
                                self.update_status_signal.emit(7, int(size / content_size * 100), "",
                                                               0)
                                # print('\r' + '[下载进度]:%s%.2f%%' % (
                                #     '>' * int(size * 50 / content_size), float(size / content_size * 100)), end=' ')
                    end = time.time()  # 下载结束时间
                    print('Done! Duration:%.2f seconds' % (end - start))  # 输出下载用时时间
                    if os.path.exists(dst):
                        print(f"start export model to path")
                        self.run_model_download(dst, Path("model/ch_PP-OCRv3_rec_infer"))
                        os.remove(dst)
                except Exception as e:
                    print(traceback.print_exc())
                    print("error, accessing url:%s exception" % url)
                    self.update_status_signal.emit(8, 0, "", 0)
            self.update_status_signal.emit(8, 0, "", 0)

    def show_install_lib_dialog(self):
        """
        显示安装库的对话框
        :param pkgs:
        :return:
        """
        print(f"wait lib installed.{self.not_install_pkgs}")
        time.sleep(2.5)
        while self.install_dialog.get_state() == 0:
            time.sleep(0.1)
        print("install_dialog click start")
        __index = 0
        success = True
        for plugin in self.not_install_pkgs:
            if not pip_plugins.install_library(__index, f'{plugin}'.split(" "),
                                               signal=self.update_status_signal):
                success = False
            __index += 1
        if not success:
            self.update_status_signal.emit(6, 0, "", 0)
            return
        else:
            # pip_plugins.move_library_to_internal()
            #reinstall numpy to the latest version
            print("reinstall numpy to the latest version")
            pip_plugins.move_library_to_internal()
            self.update_status_signal.emit(4, 0, "", 0)
