from PyQt6.QtWidgets import  QLabel
from PyQt6.QtGui import QPainter , QPen
from PyQt6.QtCore import Qt, QRect 

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
