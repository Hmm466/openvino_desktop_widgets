import os
import shutil
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
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.modelComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.modelComboBox.setObjectName("modelComboBox")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.modelComboBox)
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
        self.horizontalLayout_3.setStretch(1, 2)
        self.horizontalLayout_3.setStretch(3, 1)
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
        self.label.setText(_translate("MainWindow", "Model:"))
        self.modelComboBox.setItemText(0, _translate("MainWindow", "openai/whisper-large-v3-turbo"))
        self.modelComboBox.setItemText(1, _translate("MainWindow", "openai/whisper-large-v3"))
        self.modelComboBox.setItemText(2, _translate("MainWindow", "openai/whisper-large-v2"))
        self.modelComboBox.setItemText(3, _translate("MainWindow", "openai/whisper-large"))
        self.modelComboBox.setItemText(4, _translate("MainWindow", "openai/whisper-medium"))
        self.modelComboBox.setItemText(5, _translate("MainWindow", "openai/whisper-small"))
        self.modelComboBox.setItemText(6, _translate("MainWindow", "openai/whisper-base"))
        self.modelComboBox.setItemText(7, _translate("MainWindow", "openai/whisper-tiny"))
        self.modelComboBox.setItemText(8, _translate("MainWindow", "distil-whisper/distil-large-v2"))
        self.modelComboBox.setItemText(9, _translate("MainWindow", "distil-whisper/distil-large-v3"))
        self.modelComboBox.setItemText(10, _translate("MainWindow", "distil-whisper/distil-medium.en"))
        self.modelComboBox.setItemText(11, _translate("MainWindow", "distil-whisper/distil-small.en"))
        self.modelComboBox.setItemText(12, _translate("MainWindow", "openai/whisper-medium.en"))
        self.modelComboBox.setItemText(13, _translate("MainWindow", "openai/whisper-small.en"))
        self.modelComboBox.setItemText(14, _translate("MainWindow", "openai/whisper-base.en"))
        self.modelComboBox.setItemText(15, _translate("MainWindow", "openai/whisper-tiny.en"))
        self.label_7.setText(_translate("MainWindow", "Target"))
        self.targetCombox.setItemText(0, _translate("MainWindow", "local wav"))
        self.targetCombox.setItemText(1, _translate("MainWindow", "find audio source"))
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
        if not pip_plugins.verify_library_installed("librosa"):
            print("not install package:librosa")
            self.not_install_pkgs.append("librosa")
        if not pip_plugins.verify_library_installed("openvino_tokenizers"):
            print("not install package:openvino_genai")
            self.not_install_pkgs.append(
                "--pre -U openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly")
        if not pip_plugins.verify_library_installed("openvino_genai"):
            print("not install package:openvino_genai")
            self.not_install_pkgs.append("--pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly")
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
        self.select_model = self.modelComboBox.currentText()
        model_path = Path(f"models/whisper_asr/{self.select_model.split('/')[1]}")
        if not model_path.exists():
            print(f"local not exists :{self.select_model}.")
            self.mainBtn.setText("download model")
            # det_model.export(format="openvino", dynamic=True, half=True)
            threading.Thread(target=self.download_model_thread).start()
            self.mainBtn.setEnabled(False)
            return
        target_device = self.targetCombox.currentText()
        self.system_config = SystemConfig()
        qconfig.load("config.json", self.system_config)
        if True:
            fname, _tmp = QFileDialog.getOpenFileNames(self.window, 'Open file', './images', "*.wav")
            print(f"select image:{fname}")
            if len(fname) < 1:
                print("not selected file")
                return
            import librosa
            device = self.system_config.device
            en_raw_speech, samplerate = librosa.load(str(fname[0]), sr=16000)
            import openvino_genai
            ov_pipe = openvino_genai.WhisperPipeline(str(model_path), device=device.value)
            start = time.time()
            genai_result = ov_pipe.generate(en_raw_speech)
            print('generate:%.2f seconds' % (time.time() - start))  # 输出下载用时时间
            self.resultImageLabel.setText(f"result:{genai_result}")
        elif target_device != "find camera":
            self.camera_index = self.targetCombox.currentIndex() - 1
            self.running = True
            threading.Thread(target=self.camera_work).start()
            self.mainBtn.setText("Stop")

    update_status_signal = pyqtSignal(int, int, str, int)

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
        full_path = f"models/whisper_asr/{self.select_model.split('/')[1]}"
        success = pip_plugins.installed_model(f"export openvino --model {self.select_model} --library transformers --task automatic-speech-recognition-with-past --framework pt {full_path}", full_path, "https://hf-mirror.com")
        if os.path.exists(f"{os.getcwd()}\\py\\{full_path}\\openvino_decoder_model.xml") and os.path.exists(
                f"{os.getcwd()}\\py\\{full_path}\\openvino_decoder_model.bin"):
            print("download mode success")
            shutil.move(f"{os.getcwd()}\\py\\{full_path}", f"{os.getcwd()}\\{full_path}")
            try:
                # 删除文件夹
                shutil.rmtree(f"{os.getcwd()}\\py\\{full_path}")
            except:
                pass
        else:
            print("not model files")
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
