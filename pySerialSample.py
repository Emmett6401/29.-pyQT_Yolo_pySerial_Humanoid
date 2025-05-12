import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox  # QMessageBox 추가
from PyQt6.QtGui import QPixmap
from PyQt6 import uic
from serial_port_selector import SerialPortSelector
from motion_controller import execute_motion

# UI 파일 경로 설정
ui_file_path = os.path.join(os.path.dirname(__file__), "res", "pySerialSample.ui")

try:
    form_mainWin = uic.loadUiType(ui_file_path)[0]
except FileNotFoundError:
    print(f"Error: UI file not found at {ui_file_path}")
    sys.exit(1)

class MyWindow(QMainWindow, form_mainWin):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 실행 플래그 초기화
        self.motion_ready = False

        # 버튼 메뉴 연결
        self.pushButton_6.clicked.connect(self.open_port_selector)
        self.pushButton_1.clicked.connect(lambda: self.exeHumanoidMotion(19)) # lambda 사용 하는 것은 함수의 실행을 지연 
        self.pushButton_2.clicked.connect(lambda: self.exeHumanoidMotion(17)) # 즉 씨리얼 포트가 선택 되지 않은 상태에서        
        self.pushButton_3.clicked.connect(lambda: self.exeHumanoidMotion(18)) # 버튼 클릭 시 실행되는 함수에서 생기는 문제를 방지   
        self.pushButton_4.clicked.connect(lambda: self.exeHumanoidMotion(16))
        self.pushButton_5.clicked.connect(lambda: self.exeHumanoidMotion(22))

        # 메뉴 액션 연결
        self.actionSerial_Port.triggered.connect(self.open_port_selector)

        # 이미지 로드
        pixmap = QPixmap(os.path.join(os.path.dirname(__file__), "res", "humanoid.png"))
        self.label.setPixmap(pixmap)

    def exeHumanoidMotion(self, motion_id):
        if not self.motion_ready:
            QMessageBox.warning(self, "Motion Error", "Motion is not ready. Please select a port first.")
            return

        # 모션 실행
        execute_motion(self.lblPort.text(), motion_id, self)

    def open_port_selector(self):
        selected_port = SerialPortSelector.launch(self)
        if selected_port:
            print("선택한 포트:", selected_port)
            self.lblPort.setText(selected_port)
            # 포트가 선택되면 플래그 활성화
            self.motion_ready = True
    

# PyQt 애플리케이션 초기화
app = QApplication(sys.argv)

# 메인 윈도우 생성
window = MyWindow()
window.show()

# 애플리케이션 실행
sys.exit(app.exec_())
