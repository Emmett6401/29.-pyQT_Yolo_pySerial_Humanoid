# WebCam, YOLOv5, PyQt를 활용한 도로 시설물의 하자 탐지 프로그램
도로 시설물 탐지 프로그램

실시간 WebCam 영상과 YOLOv5 객체 탐지 모델을 기반으로, 도로 시설물의 하자 여부를 탐지하고 PyQt 기반 GUI로 시각화하는 프로그램입니다.


## 프로젝트 개요

이 프로젝트는 WebCam으로 입력된 실시간 영상을 바탕으로 YOLOv5 객체 탐지 모델을 이용하여 도로 시설물의 하자 여부를 탐지하고,

해당 정보에 대해 로그를 기록하고    

휴머노이드를 컨트롤 해서 탐지 여부를 알수 있꼬    

여러가지 기술이 있습니다. 

## 기술 스택
	•	YOLOv5: 차량 객체 탐지 모델
	•	OpenCV: 실시간 영상 처리
	•	PyQt5: GUI 설계 및 시각화
	•	Python: 전체 시스템 구현
	•	WebCam: 실시간 영상 입력 장치
 	•	pySerial: 휴머노이드 로봇을 제어 
 

## 실행 화면
![image](https://github.com/user-attachments/assets/5bb18503-98bf-4292-9b42-40243229973e)


## 설치 방법
##### • 새로운 가상환경 생성
conda create -n pyQT_yolo python=3.9

##### • 폴더 및 파일 구조 
![image](https://github.com/user-attachments/assets/b7a7d8eb-0420-4a0b-94fc-00f5e81dccd8)


##### • 1차 requirements.txt 인스톨 
pip install -3 requirement.txt

##### •  git colne
git clone http://github.cpm/ultralytics/yolo5.git

##### • 2차 requirement.txt 인스톨
cd yolov5
pip install -3 requirement.txt

