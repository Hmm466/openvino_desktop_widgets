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
import queue

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
        self.mainBtn = QtWidgets.QPushButton(self.centralwidget)
        self.mainBtn.setObjectName("mainBtn")
        self.horizontalLayout_3.addWidget(self.mainBtn)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_3.addWidget(self.lineEdit)
        self.generateButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateButton.setObjectName("generateButton")
        self.horizontalLayout_3.addWidget(self.generateButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
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
        self.verticalLayout_2.setStretch(1, 1)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.retranslateUi(MainWindow)

        self.mainBtn.clicked.connect(lambda: self.main_btn_clicked())  # type: ignore
        self.generateButton.clicked.connect(lambda: self.generate_btn_click())  # type: ignore
        self.window = MainWindow
        self.init_widget()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.mainBtn.setText(_translate("MainWindow", "load Model"))
        self.lineEdit.setText(_translate("MainWindow",
                                         "anime, masterpiece, high quality, a green snowman with a happy smiling face in the snows"))
        self.generateButton.setText(_translate("MainWindow", "generate"))

    def init_widget(self):
        self.update_status_signal.connect(self.update_status_callback)

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        self.lineEdit.setEnabled(False)
        self.generateButton.setEnabled(False)

    def generate_btn_click(self):
        print("generate_btn_click")
        text = self.lineEdit.text()
        if text != "":
            self.generate_queue.put(text)

    def main_btn_clicked(self):
        print("main btn click")
        self.not_install_pkgs = []
        if not pip_plugins.verify_library_installed("cv2"):
            print("not install package:opencv")
            self.not_install_pkgs.append("opencv-python")
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
            self.mainBtn.setEnabled(False)
            self.update_status_signal.emit(8, 0, "", 0)
            return
        self.select_model = "dreamlike-art/dreamlike-anime-1.0"
        model_path = Path(f"models/dreamlike_art/{self.select_model.split('/')[1]}")
        if not model_path.exists():
            print(f"local not exists :{self.select_model}.")
            self.mainBtn.setText("download model")
            # det_model.export(format="openvino", dynamic=True, half=True)
            threading.Thread(target=self.download_model_thread).start()
            self.mainBtn.setEnabled(False)
            return
        self.running = True
        self.mainBtn.setEnabled(False)

        self.generate_queue = queue.Queue()
        threading.Thread(target=self.generate_thread).start()

    def generate_thread(self):
        self.system_config = SystemConfig()
        qconfig.load("config.json", self.system_config)
        device = self.system_config.device
        model_path = Path(f"models/dreamlike_art/{self.select_model.split('/')[1]}")
        import openvino_genai
        import numpy as np
        import cv2
        class Generator(openvino_genai.Generator):
            def __init__(self, seed, mu=0.0, sigma=1.0):
                openvino_genai.Generator.__init__(self)
                np.random.seed(seed)
                self.mu = mu
                self.sigma = sigma

            def next(self):
                return np.random.normal(self.mu, self.sigma)

        random_generator = Generator(
            42)  # openvino_genai.CppStdGenerator can be used to have same images as C++ sample
        pipe = openvino_genai.Text2ImagePipeline(model_path, device.value)
        self.generateButton.setEnabled(True)
        self.mainBtn.setText("Stop")
        self.mainBtn.setEnabled(True)
        self.lineEdit.setEnabled(True)
        while self.running:
            item1 = self.generate_queue.get()  # 移除并返回队列中的第一个元素（A）
            prompt = item1
            start = time.time()
            print(f"start generate:{prompt}")
            image_tensor = pipe.generate(prompt, width=512, height=512, num_inference_steps=20, num_images_per_prompt=1,
                                         generator=random_generator)
            frame = image_tensor.data[0]
            img = cv2.resize(frame,
                             (self.resultImageLabel.width(), self.resultImageLabel.height()))
            self.resultImageLabel.setPixmap(QPixmap.fromImage(self.mat_to_image(img)))
            # image = Image.fromarray()

            # genai_result = ov_pipe.generate(en_raw_speech)
            print('generate:%.2f seconds' % (time.time() - start))  # 输出下载用时时间
            # self.resultImageLabel.setText(f"result:{genai_result}")
            self.generateButton.setEnabled(True)
        self.mainBtn.setEnabled(True)
        self.generateButton.setEnabled(False)
        self.lineEdit.setEnabled(False)
        self.mainBtn.setText("load Model")


    def mat_to_image(self, mat):
        if len(mat.shape) == 2:
            rows,cols = mat.shape
            bytesPerLine = cols
            return QImage(mat.data,cols,rows,bytesPerLine,QImage.Format_Indexed8)
        else:
            rows, cols,channels = mat.shape
            bytesPerLine = channels * cols
            return QImage(mat.data, cols, rows, bytesPerLine, QImage.Format_RGB888)

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
            # self.downProgressBar.setValue(index)
            pass
        elif type == 8:
            # self.downProgressBar.setVisible(False)
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
        full_path = f"models/dreamlike_art/{self.select_model.split('/')[1]}"
        success = pip_plugins.installed_model(f"export openvino --model {self.select_model} {full_path}", full_path, "https://hf-mirror.com")
        if os.path.exists(f"{os.getcwd()}\\py\\{full_path}\\tokenizer\\openvino_tokenizer.xml") and os.path.exists(
                f"{os.getcwd()}\\py\\{full_path}\\tokenizer\\openvino_tokenizer.bin"):
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
