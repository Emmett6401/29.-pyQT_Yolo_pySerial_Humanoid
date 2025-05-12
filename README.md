# WebCam, YOLOv5, PyQt를 활용한 주차장 빈자리 탐지 프로그램
주차장 빈자리 탐지 프로그램

실시간 WebCam 영상과 YOLOv5 객체 탐지 모델을 기반으로, 주차장의 빈 자리를 탐지하고 PyQt 기반 GUI로 시각화하는 프로그램입니다.


## 프로젝트 개요

이 프로젝트는 WebCam으로 입력된 실시간 영상을 바탕으로 YOLOv5 객체 탐지 모델을 이용하여 주차된 차량을 탐지하고, 해당 정보를 기반으로 주차 공간의 빈자리를 판단합니다. 사용자는 PyQt로 제작된 GUI를 통해 직관적으로 결과를 확인할 수 있습니다.

## 기술 스택
	•	YOLOv5: 차량 객체 탐지 모델
	•	OpenCV: 실시간 영상 처리
	•	PyQt5: GUI 설계 및 시각화
	•	Python: 전체 시스템 구현
	•	WebCam: 실시간 영상 입력 장치

## 실행 화면
![image](https://github.com/user-attachments/assets/d9367d28-949f-446b-a3cc-3f0ef00f1144)
![image](https://github.com/user-attachments/assets/7185e835-f18d-450d-b132-76665ecadb73)

## 설치 방법
##### • 새로운 가상환경 생성
conda create -n juso_yolo python=3.9

##### •  git colne
git clone http://github.cpm/ultralytics/yolo5.git

##### • requirement.txt 인스톨
pip install -3 requirement.txt

##### • Copilot를 이용해서 코드 완성하기
  
### PyQt로 구성된 GUI 창이 실행되며, WebCam을 통해 실시간 주차 공간 감지가 시작됩니다.
