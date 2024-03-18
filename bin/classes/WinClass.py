from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QVBoxLayout, \
     QWidget , QPushButton, QHBoxLayout , QComboBox ,QLabel,QMessageBox ,QFileDialog
from PyQt6.QtGui import QFont ,QPixmap , QImage
from PyQt6.QtCore import Qt, QSize , QBuffer
import numpy as np
import cv2

from classes.UtilEtc import  update_json_item, get_json_item
from classes.DrawMask import ImageProcessor
from classes.ImageLabelPaint import ImageLabel


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

        self.setupTopPanel()
        self.setupButton()
        self.setupComboBox()

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

        self.boximageLabel = ImageLabel()        
        self.setupSplitter()
        # 상단 패널을 메인 레이아웃에 추가
        self.main_layout.addWidget(self.top_panel)
        # QSplitter를 메인 레이아웃에 추가
        self.main_layout.addWidget(self.splitter)

        # 초기 윈도우 크기 및 위치 설정
        self.centerWindow()

        # 초기값으로 모델로드 한다,
        self.downloadYoloV8Model()
###############################################################################################    
    def setupTopPanel(self):
        # 상단 패널 설정
        self.top_panel = QWidget()
        self.top_panel.setFixedHeight(40)  # 높이를 30으로 고정
        self.top_layout = QHBoxLayout()  # 수평 레이아웃 사용
        self.top_panel.setLayout(self.top_layout)
        # 상단 패널의 배경색을 진한 회색으로 설정
        self.top_panel.setStyleSheet("background-color: #707070;")
    def setupButton(self):
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
    def setupComboBox(self):    
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
    def setupSplitter(self):
        # QSplitter 생성 및 설정
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        # QSplitter의 핸들 넓이 설정
        self.splitter.addWidget(self.boximageLabel)
        # 이 컨테이너 위젯을 오른쪽 스플리터 위젯으로 설정
        self.splitter.addWidget(self.rightWidget)

        # 스플리터의 위젯 크기를 조정합니다.
        # 예를 들어, 전체 스플리터 너비의 1/2을 왼쪽 위젯에 할당합니다.
        self.splitter.setSizes([self.width() * 1 // 2, self.width() // 2])

        self.splitter.setHandleWidth(6)
        # QSplitter의 핸들 색상을 진한 그레이색으로 설정
        self.splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: darkgray;
            }
            """
        )

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
    def openFileDialog(self):
        # 시작 디렉토리를 '././source'로 설정합니다.
        # 사용자가 .png와 .jpg 파일만 선택할 수 있도록 필터를 설정합니다.
        # options = QFileDialog.options
        fileName, _ = QFileDialog.getOpenFileName(self, "분석할 이미지를 선택하세요", 
                        "././source", "Image Files (*.png *.jpg *.gif)")
        if fileName:  # 파일이 선택되었다면
            self.displayImage(fileName)  # 선택된 파일 경로를 콘솔에 출력하거나 다른 처리를 할 수 있습니다.

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

            imgpro = ImageProcessor()

            detections =[]
            with open(self.save_path_txt, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class_id, x, y, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    detections.append((parts))



            if '-seg' in self.yoloModel or 'FastSAM' in self.yoloModel or 'sam_' in self.yoloModel:        #segmentation Model
                pmap = imgpro.draw_specific_maskes(self.yoloModel , target_arr, detections,target_class_id)
            elif '-pose' in self.yoloModel:     #person pose Mo
                pmap = imgpro.draw_pose(target_arr, detections)
            else:
                pmap = imgpro.draw_specific_boxes(target_arr, detections, target_class_id)
            
            self.resultLabel.setPixmap(pmap)
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

