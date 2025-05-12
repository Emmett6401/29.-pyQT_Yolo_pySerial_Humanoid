import os
import sys
import csv
import cv2
import torch
import pathlib
import numpy as np
from datetime import datetime
from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow, QLineEdit, QLabel, QProgressBar, QListWidget  
from PyQt6.QtCore import QTimer
from motion_controller import execute_motion
from serial_port_selector import SerialPortSelector
from serial.tools.list_ports import comports
# 경고 무시 
import warnings  # ← 추가
warnings.filterwarnings("ignore", category=FutureWarning)  # ← 경고 무시
import logging
# 로그 파일 설정
log_file = "app.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[%Y-%m-%d %H:%M:%S] %(message)s",
    encoding="utf-8"
)

class RoadDetectApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.motion_ready = False
        # RoadDetectApp2.ui 로드
        uic.loadUi("res/RoadDetectApp2.ui", self)
        self.detection_log = []  # 감지 내역 누적 저장용


        # 위젯 연결해서 객체를 직접 다룰수 있게 한다. 
        self.satellite_put = self.findChild(QLineEdit, "satellite_put")
        self.detect_put = self.findChild(QLineEdit, "detect_put")
        self.address_put = self.findChild(QLineEdit, "address_put")
        self.picture_name_put = self.findChild(QLineEdit, "picture_name_put")
        self.cam_label = self.findChild(QLabel, "cam")
        self.memo_3 = self.findChild(QLabel, "memo_3")
        self.lblPort = self.findChild(QLabel, "lblPort")
        self.lblPortLed = self.findChild(QLabel, "lblPortLed")

        self.lblPort.setText("None")  # 초기 포트 상태
        self.lblPortLed.setVisible(False)  # 포트 미연결 LED 숨기기
        self.actionAutoSelect.setChecked(False) #actionAutoSelect가 .ui 안에서 기본적으로 checked="true"로 되어 있어서,
        self.detectListWidget = self.findChild(QListWidget, "detectListWidget")


        ''' 
        만약 이 방식이 싫다면 
        # pyuic5 -o ui_main.py RoadDetectApp2.ui 이라고 터미널에서 변환 해서 직접 사용이 가능
        class RoadDetectApp(QMainWindow):
            def __init__(self):
                super().__init__()
                self.ui = Ui_MainWindow()
                self.ui.setupUi(self)
                self.ui.cam.setPixmap(...)  # 이렇게 바로 가능
        '''
        

        self.progressBar.setVisible(False)  # ProgressBar 숨기기

        # 버튼 클릭 이벤트 연결        
        self.actionAutoSelect.triggered.connect(lambda: self.open_port_selector(1))
        self.actionManualSelect.triggered.connect(lambda: self.open_port_selector(0))
        self.btnSave.clicked.connect(self.capture_picture)
        self.btnSave.setEnabled(False)

        # PosixPath 에러 방지
        if sys.platform == "win32":
            pathlib.PosixPath = pathlib.WindowsPath
        # YOLOv5 모델 로드
        '''
        여기에 오류가 많이 난다. 
        torch.hub.load()를 사용하여 YOLOv5 모델을 로드하는 부분에서 문제가 발생할 수 있습니다.
        torch.hub.load()는 PyTorch의 Hub 기능을 사용하여 모델을 다운로드하고 로드하는 방법입니다.
        방법 1. PosixPath → WindowsPath 강제 변환 (즉시 해결  이방법은 ultralytics에서도 사용하고 있다 포럼에서)
        방법 2. YOLOv5 코드 내부에서 직접 로드 (더 안정적이고 향후 유지보수 좋음)

        '''
        path = "potHolebest.pt"  # YOLOv5 모델 파일 경로
        # 방법 1        
        self.model = torch.hub.load('./yolov5', "custom", path, source="local", force_reload=True)
        # 방법 2
        '''
        YOLOv5가 Colone이 되어 있는 경우에 만 사용 가능        
        self.model = DetectMultiBackend("potHolebest.pt", device="cpu")
        '''

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.timer_active = True
        self.start_webcam()

        self.image_save_path = "image"
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

        # 마지막 사용한 포트 저장 경로        
        self.port_config_path = "last_used_port.txt"
        self.load_last_used_port()
        
        # Serial port 자동 선택 -> 마지막 사용한 포트 없을때                
        if not self.motion_ready:
            # 0.1초 후 자동 선택
            QTimer.singleShot(100, lambda: self.open_port_selector(1))
        
        logging.info("프로그램 시작")
        
    def get_daily_log_filename(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return f"detection_{today}.txt"
    
    def closeEvent(self, event):
        self.cap.release()
        logging.info("프로그램 종료")
        # 감지 로그 저장
        filename = self.get_daily_log_filename()
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now():%H:%M:%S}] 종료 시 저장\n")
            for line in self.detection_log:
                f.write(line + "\n")
        
        super().closeEvent(event)

    def load_last_used_port(self):
        if os.path.exists(self.port_config_path):
            with open(self.port_config_path, "r") as f:
                last_port = f.read().strip()
                ports = [port.device for port in comports()]
                if last_port in ports:
                    self.lblPort.setText(last_port)
                    self.motion_ready = True
                    self.lblPortLed.setVisible(True)
                    self.lblPortLed.setStyleSheet("background-color: green;")
                    print(f"최근 사용한 포트 자동 설정: {last_port}")

    def open_port_selector(self, nMethod): 
        if nMethod == 0:
            # 수동 포트 선택            
            selected_port = SerialPortSelector.launch(self)
            if selected_port:            
                self.lblPort.setText(selected_port)
                # 포트가 선택되면 플래그 활성화
                self.motion_ready = True
                self.lblPortLed.setVisible(True)  # 포트 연결 LED 표시
                self.lblPort.setText(selected_port)
                self.lblPortLed.setStyleSheet("background-color: green;")  # 포트 연결 LED 표시
        else:
            selector = SerialPortSelector(self)
            selector.auto_select_cp2104_port()  # cp2104 포트를 자동으로 선택
            selected_port = selector.selected_port
            if selected_port:
                print("자동으로 선택된 포트:", selected_port)
                self.lblPort.setText(selected_port)
                self.motion_ready = True
                self.lblPortLed.setVisible(True)
                self.lblPortLed.setStyleSheet("background-color: green;")
            else:
                print("cp2104 포트를 찾을 수 없습니다.")
        if selected_port:
            logging.info(f"포트 연결 성공: {selected_port}")
            with open(self.port_config_path, "w") as f:
                f.write(selected_port)
    def exeHumanoidMotion(self, motion_id):
            if not self.motion_ready:
                print(f"Motion Error", "Motion is not ready. Please select a port first.")
                return            
            # 모션 실행
            execute_motion(self.lblPort.text(), motion_id, self)

    def start_webcam(self):
        # 웹캠 화면을 QLabel에 표시
        def update_frame():
            if self.timer_active:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame

                    # YOLOv5 모델로 객체 감지
                    results = self.model(frame)
                    annotated_frame = np.squeeze(results.render())

                    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.cam_label.setPixmap(pixmap.scaled(self.cam_label.width(), self.cam_label.height()))

                    # 감지된 객체 리스트 초기화
                    self.detectListWidget.clear()

                    # 감지 결과 표시
                    detections = results.xyxy[0].cpu().numpy()
                    pothole_detected = False

                    for detection in detections:
                        x1, y1, x2, y2, conf, cls = detection
                        class_name = self.model.names[int(cls)]
                        conf_percent = f"{conf * 100:.1f}%"
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        satellite = self.satellite_put.text().strip()  # QLineEdit에서 좌표 가져오기
                        if not satellite:
                            satellite = "좌표 없음"
                        label = f"{class_name} ({conf_percent}) 위치=({int(x1)}, {int(y1)}) 크기=({width}x{height}) 좌표={satellite}"
                        self.detection_log.append(label)  # ✅ 실제 로그에 누적
                        self.detectListWidget.addItem(label)  # ✅ 화면에 출력

                        # 100개 넘으면 자동 저장 후 초기화
                        if len(self.detection_log) >= 100:
                            filename = self.get_daily_log_filename()
                            with open(filename, "a", encoding="utf-8") as f:
                                f.write(f"\n[{datetime.now():%H:%M:%S}] 100개 이상 감지됨\n")
                                for line in self.detection_log:
                                    f.write(line + "\n")                                
                            self.detection_log.clear()  # 다시 누적 시작

                        if class_name == "pothole" and conf > 0.5:
                            self.className = class_name
                            pothole_detected = True

                    self.btnSave.setEnabled(pothole_detected)
            

        # 타이머를 사용하여 주기적으로 화면 업데이트
        self.timer = self.startTimer(30)  # 30ms마다 업데이트
        self.timerEvent = lambda event: update_frame()

    def capture_picture(self):
        # ProgressBar 표시
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)

        # 웹캠 화면 멈추기
        self.timer_active = False
        QApplication.processEvents()  # UI 업데이트

        if self.className == "pothole":
            print("포트홀 감지됨")
        else:
            print("포트홀 감지되지 않음")
            self.timer_active = True
            self.progressBar.setVisible(False)  # ProgressBar 숨기기
            return

        if self.current_frame is not None:
            # 사진 저장
            self.progressBar.setValue(15)  # 진행률 업데이트
            QApplication.processEvents()  # UI 업데이트
            picture_name = datetime.now().strftime(f"{self.className}_%Y%m%d_%H%M%S.jpg")  # 현재 날짜와 시간으로 파일 이름 생성
            file_name = os.path.join(self.image_save_path, picture_name)
            cv2.imwrite(file_name, self.current_frame)
            
            print(f"사진이 저장되었습니다: {file_name}")            
            
            # 저장된 파일 경로를 picture_name_put(QLineEdit)에 표시
            self.picture_name_put.setText(file_name.replace("\\", "/"))
            self.progressBar.setValue(30)  # 진행률 업데이트
            

            if self.motion_ready:
                self.progressBar.setValue(50)  # 진행률 업데이트
                # 로봇 모션
                self.exeHumanoidMotion(19)  # 예시로 19번 모션 실행
                # 감지된 객체 저장                
                self.save_data()  
                self.progressBar.setValue(75)  # 진행률 업데이트                

            # 로봇 동작 완료를 확인한 후 ProgressBar를 숨기도록 수정
            QTimer.singleShot(1500, lambda: self.progressBar.setValue(100))  # 1.5초 대기 후 진행 완료
            QTimer.singleShot(1500, lambda: self.progressBar.setVisible(False))  # 1.5초 대기 후 ProgressBar 숨기기
            self.progressBar.setVisible(False)  # ProgressBar 숨기기
            QApplication.processEvents()  # UI 업데이트        
                
        self.timer_active = True  # 웹캠 다시 시작

    def save_data(self):
        # 입력된 데이터를 CSV 파일에 저장
        satellite = self.satellite_put.text()
        roadDetect = self.detect_put.text()
        address = self.address_put.text()
        image_file = self.picture_name_put.text()  # picture_put에서 파일 경로 가져오기

        if satellite and roadDetect and os.path.exists(image_file):  # 이름, 전화번호, 사진이 있는지 확인
            file_exists = os.path.isfile("RoadDetect_data.csv")
            try:
                with open("RoadDetect_data.csv", "a", newline="", encoding="utf-8-sig") as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["위성좌표", "도로 하자 종류", "도로명 주소", "사진"])
                    
                    # 감지된 객체 정보 추가
                    detected_objects = []
                    results = self.model(cv2.imread(image_file))
                    detections = results.xyxy[0].cpu().numpy()
                    for detection in detections:
                        cls = int(detection[5])
                        detected_objects.append(self.model.names[cls])
                    
                    writer.writerow([satellite, roadDetect, address, image_file])
                print(f"저장된 데이터: address={satellite}, roadDetect={roadDetect}, address={address}, 사진={image_file}, 감지된 객체={', '.join(detected_objects)}")
            except Exception as e:
                print(f"데이터 저장 중 오류 발생: {e}")
        else:
            print("satellite, roadDetect, address 필수 항목입니다.")

    def closeEvent(self, event):
        # 종료 시 웹캠 해제
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    parking_app = RoadDetectApp()
    parking_app.show()
    sys.exit(app.exec())