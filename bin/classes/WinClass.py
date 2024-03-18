from PyQt6.QtWidgets import QFileDialog ,QApplication, QMainWindow, QSplitter, QTextEdit, QVBoxLayout, \
     QWidget , QPushButton, QHBoxLayout , QComboBox ,QLabel,QMessageBox
from PyQt6.QtGui import QPainter, QPen, QPixmap ,QImage , QFont
from PyQt6.QtCore import Qt, QSize, QRect , QBuffer

import numpy as np
import cv2
from classes.UtilEtc import  update_json_item, get_json_item


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # yolo model
        self.model = None
        self.copiedImg = QLabel()

        # 결과 이미지를 파일로 저장
        self.save_path_img = '././result/result.jpg'  # 결과를 저장할 파일 경로
        self.save_path_txt = '././result/result.txt'  # 결과를 저장할 파일 경로
        self.save_json_file = './bin/setting.json'
        # 저장된 모델 가저오기
        self.yoloModel = get_json_item(self.save_json_file,'YoloModel') 
        self.modelPath = get_json_item(self.save_json_file,'ModelPath') 

        # 윈도우 타이틀 설정
        self.setWindowTitle("YoloV8 Analysis Tool")
        # 상단 패널 설정
        self.top_panel = QWidget()
        self.top_panel.setFixedHeight(40)  # 높이를 30으로 고정
        self.top_layout = QHBoxLayout()  # 수평 레이아웃 사용
        self.top_panel.setLayout(self.top_layout)
        # 상단 패널의 배경색을 진한 회색으로 설정
        self.top_panel.setStyleSheet("background-color: #707070;")

        # 상단 패널에 버튼 추가
        self.button1 = QPushButton("이미지 선택")
        self.button1.setStyleSheet("background-color: #d3d3d3;")  # 연한 회색으로 배경색 설정        
        self.button1.clicked.connect(self.openFileDialog) # 버튼1 클릭 시그널에 메소드 연결

        self.button2 = QPushButton("이미지 분석")
        self.button2.setStyleSheet("background-color: #d3d3d3;")  # 연한 회색으로 배경색 설정
        self.button2.clicked.connect(self.processAndDisplayImage)

        self.button3 = QPushButton("Load YoloV8 model")
        self.button3.setStyleSheet("background-color: #d3d3d3;")  # 연한 회색으로 배경색 설정
        self.button3.clicked.connect(self.downloadYoloV8Model)

        self.top_layout.addWidget(self.button1)
        self.top_layout.addWidget(self.button2)
        self.top_layout.addWidget(self.button3)

        # 상단 패널에 콤보박스 추가
        self.comboBoxModel = QComboBox()
        self.comboBoxModel.setStyleSheet("background-color: #d3d3d3;")  # 연한 회색으로 배경색 설정
        self.comboBoxModel.addItem("yolov8n.pt")
        self.comboBoxModel.addItem("yolov8s.pt")
        self.comboBoxModel.addItem("yolov8m.pt")
        self.comboBoxModel.addItem("yolov8l.pt")
        self.comboBoxModel.addItem("yolov8x.pt")
        self.comboBoxModel.addItem("yolov8n-seg.pt")
        self.comboBoxModel.addItem("yolov8s-seg.pt")
        self.comboBoxModel.addItem("yolov8m-seg.pt")
        self.comboBoxModel.addItem("yolov8l-seg.pt")
        self.comboBoxModel.addItem("yolov8x-seg.pt")
        self.comboBoxModel.addItem("yolov8n-pose.pt")
        self.comboBoxModel.addItem("yolov8s-pose.pt")
        self.comboBoxModel.addItem("yolov8m-pose.pt")
        self.comboBoxModel.addItem("yolov8l-pose.pt")
        self.comboBoxModel.addItem("yolov8x-pose.pt")
        self.comboBoxModel.addItem("yolov8x-pose-p6.pt")
        self.comboBoxModel.addItem("sam_b.pt") # SAM base
        self.comboBoxModel.addItem("sam_l.pt") # SAM large
        self.comboBoxModel.addItem("FastSAM-s.pt") # Fasr SAM
        self.comboBoxModel.addItem("FastSAM-x.pt") # Fasr SAM
        self.comboBoxModel.addItem("yolo_nas_s.pt") # YOLO-NAS-s
        self.comboBoxModel.addItem("yolo_nas_m.pt") # YOLO-NAS-m
        self.comboBoxModel.addItem("yolo_nas_l.pt") # YOLO-NAS-l
        self.comboBoxModel.addItem("rtdetr-l.pt") # 실시가 감지 트랜스포머
        self.comboBoxModel.addItem("rtdetr-x.pt") # YOLO-NAS-l


        # self.comboBoxModel.setCurrentText(self.yoloModel)
        if self.yoloModel in [self.comboBoxModel.itemText(i) for i in range(self.comboBoxModel.count())]:
            self.comboBoxModel.setCurrentText(self.yoloModel)
        else:
            print(f"Warning: Model {self.yoloModel} is not available.")

        self.comboBoxModel.currentIndexChanged.connect(self.on_comboboxModel_changed)
        self.top_layout.addWidget(self.comboBoxModel)

        # 상단 패널에 콤보박스 추가
        self.comboBox = QComboBox()
        self.comboBox.setStyleSheet("background-color: #d3d3d3;")  # 연한 회색으로 배경색 설정
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)
        self.comboBox.addItem("label: count")
        self.top_layout.addWidget(self.comboBox)
        
        # 메인 위젯 및 레이아웃 설정
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # 선택된 항목을 큰 글씨로 보여줄 QLabel 생성
        self.selectedItemLabel = QLabel("Model: " + self.yoloModel + " Selected Object: None")
        self.selectedItemLabel.setFont(QFont('Arial', 22))  # 글꼴과 크기 설정
        self.selectedItemLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)  # 텍스트 중앙 정렬

        self.resultLabel  = QLabel()
        # 오른쪽 위젯을 위한 레이아웃을 생성하고, 선택된 항목 레이블과 결과 레이블을 추가
        self.rightLayout = QVBoxLayout()
        self.rightLayout.addWidget(self.selectedItemLabel)
        self.rightLayout.addWidget(self.resultLabel)
        # QVBoxLayout을 포함하는 컨테이너 위젯 생성
        self.rightWidget = QWidget()
        self.rightWidget.setLayout(self.rightLayout)

        # QSplitter 생성 및 설정
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        # QSplitter의 핸들 넓이 설정
        self.splitter.setHandleWidth(6)
        # QSplitter의 핸들 색상을 진한 그레이색으로 설정
        self.splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: darkgray;
            }
            """
        )
        self.boximageLabel = ImageLabel()
        self.splitter.addWidget(self.boximageLabel)
        # 이 컨테이너 위젯을 오른쪽 스플리터 위젯으로 설정
        self.splitter.addWidget(self.rightWidget)

        # 스플리터의 위젯 크기를 조정합니다.
        # 예를 들어, 전체 스플리터 너비의 1/2을 왼쪽 위젯에 할당합니다.
        self.splitter.setSizes([self.width() * 1 // 2, self.width() // 2])

        # 상단 패널을 메인 레이아웃에 추가
        self.main_layout.addWidget(self.top_panel)
        # QSplitter를 메인 레이아웃에 추가
        self.main_layout.addWidget(self.splitter)

        # 초기 윈도우 크기 및 위치 설정
        self.centerWindow()

        # 초기값으로 모델로드 한다,
        self.downloadYoloV8Model()
###############################################################################################    
    def centerWindow(self):
        # 스크린 사이즈 가져오기
        screen = QApplication.primaryScreen().geometry()
        
        # 윈도우 사이즈를 스크린의 2/3로 설정
        width = screen.width() * 2 / 3
        height = screen.height() * 2 / 3
        
        # 윈도우 최소 사이즈 설정 (선택적)
        self.setMinimumSize(QSize(int(width), int(height)))

        # 윈도우를 스크린 중앙에 위치
        self.setGeometry(
            int((screen.width() - width) / 2), 
            int((screen.height() - height) / 2),
            int(width),
            int(height)
        )
###############################################################################################
    def on_combobox_changed(self, index):
        # 콤보박스에서 선택된 아이템 텍스트 가져오기
        item_text = self.comboBox.currentText()
        # 'Car: 17' 형식에서 레이블 이름 분리하기
        label_name = item_text.split(':')[0]  # 'Car' 부분만 가져옴
        if label_name !='' and label_name != 'label':
            # pixmap = self.resultLabel.pixmap()
            pixmap = self.copiedImg.pixmap()
            
            if pixmap is not None:
                image = pixmap.toImage()
                # QImage를 NumPy 배열로 변환
                ptr = image.bits()
                ptr.setsize(image.sizeInBytes())
                target_arr = np.array(ptr).reshape(image.height(), image.width(), 4)  # RGBA 포맷
                # OpenCV에서 사용하기 위해 RGB 포맷으로 변환
                target_arr = cv2.cvtColor(target_arr, cv2.COLOR_RGBA2RGB)
            else:
                print('이미지를 확보하지 못했습니다.')
                return

            # 주어진 레이블 이름에 해당하는 클래스 번호 찾기
            target_class_id = None
            if self.yoloModel.startswith('sam_'):
                target_class_id = 0
            else:
                for class_id, name in self.model.names.items():  # 수정: enumerate 사용
                    if name == label_name:
                        target_class_id = class_id
                        break

            if target_class_id is None:
                print(f"Label '{label_name}' not found in class names.")
                return
            
            self.selectedItemLabel.setText(f"Model: {self.yoloModel} Selected Object: {item_text}")

            if '-seg' in self.yoloModel or 'FastSAM' in self.yoloModel or 'sam_' in self.yoloModel:        #segmentation Model
                self.draw_specific_maskes(self.save_path_txt , target_arr,target_class_id)
            elif '-pose' in self.yoloModel:     #person pose Model
                self.draw_pose(self.save_path_txt , target_arr)
            else:
                self.draw_specific_boxes(self.save_path_txt , target_arr, target_class_id)
###############################################################################################
    def on_comboboxModel_changed(self, index):
        # 콤보박스에서 선택된 아이템 텍스트 가져오기
        self.yoloModel = self.comboBoxModel.currentText()
        self.button3.setEnabled(True) #다운로드 버튼이 가능한 상태로 변경
###############################################################################################
    def displayImage(self, imagePath):
        # 이미지 로드
        pixmap = QPixmap(imagePath)
        self.boximageLabel.setOriginalPixmap(pixmap.scaled(self.splitter.width() // 2, 
                                                self.splitter.height(), 
                                                Qt.AspectRatioMode.KeepAspectRatio, 
                                                Qt.TransformationMode.SmoothTransformation))

###############################################################################################
    def openFileDialog(self):
        # 시작 디렉토리를 '././source'로 설정합니다.
        # 사용자가 .png와 .jpg 파일만 선택할 수 있도록 필터를 설정합니다.
        # options = QFileDialog.options
        fileName, _ = QFileDialog.getOpenFileName(self, "분석할 이미지를 선택하세요", 
                        "././source", "Image Files (*.png *.jpg *.gif)")
        if fileName:  # 파일이 선택되었다면
            self.displayImage(fileName)  # 선택된 파일 경로를 콘솔에 출력하거나 다른 처리를 할 수 있습니다.
###############################################################################################
    def processAndDisplayImage(self):
        # 이미지 영역이 선택되었는지 확인
        if not self.boximageLabel.selectionRect.isNull():
            # 선택된 이미지 영역 가져오기
            selectedArea = self.boximageLabel.selectionRect
            # 원본 이미지에서 선택된 영역의 이미지를 추출
            croppedQImage = self.boximageLabel.originalPixmap.copy(selectedArea).toImage()

            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(croppedQImage)
            # Display the QPixmap in your QLabel
            # self.resultLabel.setPixmap(pixmap)
            self.copiedImg.setPixmap(pixmap)

            # QImage를 NumPy 배열로 변환
            buffer = QBuffer()
            buffer.open(QBuffer.OpenModeFlag.ReadWrite)
            croppedQImage.save(buffer, "PNG")
            buffer.close()  # 사용한 버퍼는 닫아주는 것이 좋습니다.
            # QImage에서 바로 데이터를 가져와 NumPy 배열로 변환
            pil_im = croppedQImage.convertToFormat(QImage.Format.Format_RGBA8888)
            ptr = pil_im.constBits()
            ptr.setsize(pil_im.sizeInBytes())  # 중요: 메모리의 정확한 크기를 설정합니다.
            croppedQImage = np.array(ptr, dtype=np.uint8).reshape(pil_im.height(), pil_im.width(), 4)
            # BGR 형식으로 변환 (YOLO 모델은 RGB가 아닌 BGR을 사용)
            croppedImage = cv2.cvtColor(croppedQImage, cv2.COLOR_RGBA2BGR)

            # 모델이 로드되었는지 확인
            if not self.model:
                print("Model is not loaded")
                return
            if self.yoloModel.startswith('sam_'):
                reply = QMessageBox.warning(self,
                                                "작업경고","무거운 가중치가 있는모델로 예측 합니다 \n 작업시간이 4분 이상 소요됩니다.",
                                                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,   
                                                QMessageBox.StandardButton.Cancel  )                                 
                if reply == QMessageBox.StandardButton.Ok:
                    results = self.model(croppedImage)
                    # 예측 결과에서 클래스 ID 추출
                    # print('='*100)
                    # print(type(results))
                    # print('='*100)
                    # import torch
                    # if isinstance(results, (np.ndarray, torch.Tensor)):
                    #     print(results.shape)
                    # else:
                    #     # results가 다른 타입인 경우, 추가적인 처리 방법을 고려해야 할 수 있음
                    #     print("results는 넘파이 배열이나 파이토치 텐서가 아닙니다.")
                    # print(results )
                    # print('='*100)
                    for r in results:
                        print(r.boxes.cls , r.boxes.id)
                else:
                    return
            else:
                results = self.model(croppedImage)

            results[0].save(self.save_path_img)  # 결과 이미지 파일로 저장
            # 결과를 저장할 파일이 존재하는 경우, 내용을 비움
            open(self.save_path_txt, 'w').close()
            results[0].save_txt(self.save_path_txt)  # 결과 이미지 파일로 저장

            self.summarize_detections(self.save_path_txt)
###############################################################################################
    def summarize_detections(self,file_path):
        # 클래스별로 발생한 횟수를 저장할 딕셔너리
        # 콤보박스 아이템 초기화
        self.comboBox.clear()

        # 파일 읽기
        class_counts = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if self.yoloModel.startswith('sam_'):
            self.comboBox.addItem(f"objects:{len(lines)}")
            # if not hasattr(self.model, 'names'):
            #     self.model.names = {}
            # self.model.names[0] = 'objects'
            return        
            
        # 데이터 분리 및 첫 번째 열(클래스 ID) 기준으로 정렬
        detections = [line.strip().split() for line in lines]
        detections.sort(key=lambda x: int(x[0]))  # 첫 번째 열(클래스 ID) 기준으로 정렬

        # 클래스별 카운트 집계
        for det in detections:
            class_id = int(det[0])  # 첫 번째 열은 클래스 ID
            # 클래스 ID를 기반으로 카운트 증가
            if class_id in class_counts:
                class_counts[class_id] += 1
            else:
                class_counts[class_id] = 1

        # 새로운 결과를 콤보박스에 추가
        for class_id, count in class_counts.items():
            # 클래스 ID를 레이블 이름으로 변환
            label_name = self.model.names[class_id] if class_id < len(self.model.names) else "Unknown"
            # 콤보박스에 결과 추가
            self.comboBox.addItem(f"{label_name}:{count}")
###############################################################################################
    def draw_specific_boxes(self, results_path , target_arr,target_class_id):
        # 결과 파일 읽기
        with open(results_path, 'r') as f:
            lines = f.readlines()

        # 각 객체에 대한 바운딩 박스 그리기
        color = [int(c) for c in np.random.choice(range(256), size=3)]
        cnt = 0
        for line in lines:
            parts = line.strip().split()
            class_id, x, y, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            if class_id == target_class_id:
                cnt = cnt + 1
                x1, y1 = int((x - bw / 2) * target_arr.shape[1]), int((y - bh / 2) * target_arr.shape[0])
                x2, y2 = int((x + bw / 2) * target_arr.shape[1]), int((y + bh / 2) * target_arr.shape[0])
                cv2.rectangle(target_arr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(target_arr, str(cnt), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # NumPy 배열 이미지를 QImage로 변환
        # OpenCV는 BGR 포맷을 사용하므로, QImage에서 사용할 수 있게 RGB로 다시 변환해야 합니다.
        arr_rgb = cv2.cvtColor(target_arr, cv2.COLOR_BGR2RGB)
        height, width, channels = arr_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(arr_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(q_img)
        # QPixmap을 QLabel에 설정
        self.resultLabel.setPixmap(pixmap)
###############################################################################################
    def draw_specific_maskes(self, results_path , target_arr,target_class_id):
        # 결과 파일 읽기
        with open(results_path, 'r') as f:
            lines = f.readlines()

        height, width = target_arr.shape[:2]

        # 각 객체에 대한 Mask 그리기
        cnt = 0
        for line in lines:
            parts = line.strip().split()
            if 'sam_' in self.yoloModel:   # sam_ 모댈은 첫 글자를 발생한 object seq로 발생해서 0으로 치환  
                class_id = 0
            else:
                class_id = int(parts[0])

            if class_id == target_class_id:
                cnt = cnt + 1
                color = [int(c) for c in np.random.choice(range(100,256), size=3)]
                mask_coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2) # 나머지는 mask 좌표임
                # 마스크 좌표를 이미지의 크기에 맞게 조정
                mask_coords[:, 0] = mask_coords[:, 0] * width
                mask_coords[:, 1] = mask_coords[:, 1] * height
                mask_coords = np.int32([mask_coords])                
                # 마스크를 반투명 이미지에 그리기
                target_arr = self.draw_transparent_poly(target_arr, mask_coords, color=color, alpha=0.7)
                # 다각형 그리기 (여기서는 경계선만 그립니다)
                cv2.polylines(target_arr, [mask_coords], isClosed=True, color=(0, 0, 0), thickness=2)

                text_position = (int(mask_coords[0][-1][0]), int(mask_coords[0][-1][1]))  # 마지막 좌표를 텍스트 위치로 사용
                cv2.putText(target_arr, str(cnt), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        arr_rgb = cv2.cvtColor(target_arr, cv2.COLOR_BGR2RGB)
        height, width, channels = arr_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(arr_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(q_img)
        # QPixmap을 QLabel에 설정
        self.resultLabel.setPixmap(pixmap)
################################################################################
    # 반투명한 마스크를 이미지에 그리기 위한 함수
    def draw_transparent_poly(self, img, mask_coords, color, alpha=0.5):
        # 새로운 마스크 이미지 생성 (입력 이미지와 같은 크기)
        overlay = img.copy()
        output = img.copy()
        # 마스크에 폴리곤(다각형)을 그리기 - 이 폴리곤은 불투명
        cv2.fillPoly(overlay, mask_coords, color=color, lineType=cv2.LINE_AA)
        # 원본 이미지와 새로운 마스크 이미지를 알파 값으로 혼합
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
###############################################################################################    
    def draw_pose(self,results_path , target_arr):
        # 결과 파일 읽기
        with open(results_path, 'r') as f:
            lines = f.readlines()

        # 이미지의 크기를 얻기
        height, width = target_arr.shape[:2]
        limbs_info = {
                # (limb_index): (B, G, R) 색상 코드
            (0, 1) : (255,0,0), # nose -> left-eye
            (1, 3) : (255,0,0), # left-eye -> left-ear
            (2, 3) : (255,0,0), # right-eye ->left-ear 
            (0, 2) : (255,0,0), # nose -> right-eye
            (2, 4) : (255,0,0), # right-eye -> right-ear
            (3, 5) : (255,0,0), # left-ear -> left-shoulder
            (4, 6) : (255,0,0), # right-ear -> right-shoulder
            (5, 6) : (0,255,255), # left-shoulder -> right-shoulder
            (6, 12): (0,255,255), # right-shoulder -> right-hip
            (12, 11): (0,255,255), # right-hip -> left-hip
            (11, 5): (0,255,255), # left-hip -> left-shoulder
            (5, 7): (0,0,255), # left-shoulder -> left-elbow
            (7, 9): (0,0,255), # left-elbow -> left-wrist
            (6, 8): (0,0,255), # right-shoulder -> right-elbow
            (8, 10): (0,0,255), # right-elbow -> right-wrist
            (12, 14): (0,0,255), # right-hip -> right-knee
            (14, 16): (0,0,255), # right-knee -> right-ankle
            (11, 13): (0,0,255), # left-hip -> left-knee
            (13, 15): (0,0,255) # left-knee -> left-ankle
            # 이하 생략, 각 연결에 대해 다른 색상 코드 추가 가능
        }
            # 더 많은 연결이 필요할 수 있음

        # 사람의 중심 좌표 계산
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # 데이터가 충분하지 않으면 무시
                continue
            try:
                # 문자열을 실수로 변환 후, 이미지 크기에 맞게 조정
                center_x, center_y = float(parts[3]) * width, float(parts[4]) * height
            except ValueError:  # 변환 실패 시 다음 줄로 넘어감
                print(f"Cannot convert: {parts[1]}, {parts[2]}")
                continue
       
            # 각 관절 위치를 저장하는 리스트 생성
            cnt = 0
            keypoints = []
            for i in range(5, len(parts), 3):
                # 각 관절의 x, y 좌표
                x, y = int(float(parts[i]) * width), int(float(parts[i+1]) * height)
                keypoints.append((x, y))
                # 이미지에 각 관절 그리기
                cv2.circle(target_arr, (x, y), 5, (0, 255, 0), thickness=-1)
                cv2.putText(target_arr, str(cnt), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cnt = cnt + 1
            # 두 관절 사이를 연결하는 선 그리기 (여기서는 예시로 몇 개의 관절만 연결합니다)
            for (start_index, end_index) , color  in limbs_info.items():
                start, end = keypoints[start_index], keypoints[end_index]
                # start 또는 end 값이 0이면 라인을 그리지 않음
                if start == (0, 0) or end == (0, 0):
                    continue
                cv2.line(target_arr, start, end, color, thickness=2)        # # NumPy 배열 이미지를 QImage로 변환
            # OpenCV는 BGR 포맷을 사용하므로, QImage에서 사용할 수 있게 RGB로 다시 변환해야 합니다.
        arr_rgb = cv2.cvtColor(target_arr, cv2.COLOR_BGR2RGB)
        height, width, channels = arr_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(arr_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(q_img)
        # QPixmap을 QLabel에 설정
        self.resultLabel.setPixmap(pixmap)
###############################################################################################
    def downloadYoloV8Model(self):
        model_path = self.modelPath + self.yoloModel
        QMessageBox.information(self,"Model download",model_path+" 모델을 다운로드 하고 해석 중입니다.......")
    
        from ultralytics import FastSAM, SAM, NAS ,YOLO , RTDETR
        import ultralytics
        
        ultralytics.checks()
        if self.yoloModel.startswith('yolov8'):
            # load Yolo model
            self.model = YOLO(model_path)
        elif self.yoloModel.startswith('sam_'):
            self.model = SAM(model_path)
        elif self.yoloModel.startswith('FastSAM'):
            self.model = FastSAM(model_path)
        elif self.yoloModel.startswith('yolo_nas'):
            self.model = NAS(model_path)
        elif self.yoloModel.startswith('rtdetr-'):
            self.model = RTDETR(model_path)
        # 모델이 제대로 로드되었는지 확인
        if self.model:
            print(self.model.info(detailed = False))
           # print(f"Model loaded successfully. Number of classes: {len(self.model.names)}")
           # print(self.model.names)
            # 가장 최근 로드한 모델은 setting.json 파일에 저장 한다
            update_json_item(self.save_json_file,'YoloModel',self.yoloModel)
            # 모델이 성공적으로 로드되었다면 다운로드 버튼을 비활성화
            self.button3.setEnabled(False)
        else:
            # 모델 로드 실패 시 메시지 표시
            QMessageBox.warning(self, "Model Load Failed", "모델 로드에 실패했습니다.")            
    
###############################################################################################
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.originalPixmap = None  # 원본 이미지를 저장할 변수
        self.selectionRect = QRect()
        self.isSelecting = False

    def setOriginalPixmap(self, pixmap):
        self.originalPixmap = pixmap

        # 이미지의 원본 크기를 가져옵니다.
        imgRect = QRect(self.rect().topLeft(), self.originalPixmap.size())        
        self.selectionRect = QRect() # 새 이미지를 설정할 때마다 사각형 선택 영역도 초기화
        self.selectionRect = imgRect 
        self.update()  # 이미지를 변경할 때 화면을 다시 그립니다.

    def mousePressEvent(self, event):
        # if event.button() == Qt.MouseButton.LeftButton:
        if self.originalPixmap and self.originalPixmap.rect().contains(event.pos()):
            self.isSelecting = True
            self.selectionRect.setTopLeft(event.pos())
            self.selectionRect.setBottomRight(event.pos())

    def mouseMoveEvent(self, event):
        if self.isSelecting and self.originalPixmap.rect().contains(event.pos()):
            self.selectionRect.setBottomRight(event.pos())
            self.update()  # Force the label to be redrawn

    def mouseReleaseEvent(self, event):
        # if event.button() == Qt.MouseButton.LeftButton:
        if self.isSelecting:
            self.isSelecting = False
            # 영역을 이미지 내로 제한
            rect = self.originalPixmap.rect().intersected(self.selectionRect)
            self.selectionRect = rect
            self.update()
        if not self.selectionRect.isNull():
            parent = self.parent()
            while parent is not None and not isinstance(parent, MainWindow):
                parent = parent.parent()  # 부모를 계속 타고 올라가 MainWindow를 찾음

            if parent and isinstance(parent, MainWindow):
                parent.processAndDisplayImage()  # MainWindow의 processAndDisplayImage 호출
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QPainter(self)
        if self.originalPixmap:
            # QLabel의 전체 영역에 원본 이미지를 그립니다.
            # painter.drawPixmap(self.rect(), self.originalPixmap)

            # 이미지를 QLabel의 전체 영역에 맞추지 않고 원본 크기로 그립니다.
            painter.drawPixmap(self.rect().topLeft(), self.originalPixmap)
        if not self.selectionRect.isNull():
            # 사용자가 선택한 영역에 사각형을 그립니다.
            self.boxDraw(painter,self.selectionRect)
    def boxDraw(self,bPainter,bRect):
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(2)
        bPainter.setPen(pen)
        bPainter.drawRect(bRect)
###############################################################################################

