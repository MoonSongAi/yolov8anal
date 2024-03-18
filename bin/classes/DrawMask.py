import numpy as np
import cv2
from PyQt6.QtGui import QPainter, QPen, QPixmap ,QImage , QFont

class ImageProcessor:
    def __init__(self):
        # 초기화 로직이 필요한 경우 여기에 추가
        pass

    def draw_specific_boxes(self,target_arr, detections, target_class_id):
        # 각 객체에 대한 바운딩 박스 그리기
        cnt = 0
        color = [int(c) for c in np.random.choice(range(256), size=3)]
        for det in detections:
            class_id, x, y, bw, bh = det
            if class_id == target_class_id:
                cnt += 1
                x1, y1 = int((x - bw / 2) * target_arr.shape[1]), int((y - bh / 2) * target_arr.shape[0])
                x2, y2 = int((x + bw / 2) * target_arr.shape[1]), int((y + bh / 2) * target_arr.shape[0])
                cv2.rectangle(target_arr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(target_arr, str(cnt), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        pmap = self.convert_cv2QImage(target_arr)
        return pmap
##########################################################################################################
    def draw_pose(self,target_arr , detections):
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
        

        for det in detections:
            cnt = 0
            keypoints = []
            for i in range(5, len(det), 3):
                # 각 관절의 x, y 좌표
                x, y = int(float(det[i]) * width), int(float(det[i+1]) * height)
                keypoints.append((x, y))
                print( x,y)
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
        pmap = self.convert_cv2QImage(target_arr)
        return pmap

    def draw_specific_maskes(self, model, target_arr,detections,target_class_id):
        height, width = target_arr.shape[:2]

        # 각 객체에 대한 Mask 그리기
        cnt = 0
        for det in detections:
            if 'sam_' in model:   # sam_ 모댈은 첫 글자를 발생한 object seq로 발생해서 0으로 치환  
                class_id = 0
            else:
                class_id = int(det[0])

            if class_id == target_class_id:
                cnt = cnt + 1
                color = [int(c) for c in np.random.choice(range(100,256), size=3)]
                mask_coords = np.array(det[1:], dtype=np.float32).reshape(-1, 2) # 나머지는 mask 좌표임
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

        pmap = self.convert_cv2QImage(target_arr)
        return pmap


    def draw_transparent_poly(self, img, mask_coords, color, alpha=0.5):
        overlay = img.copy()
        output = img.copy()
        cv2.fillPoly(overlay, mask_coords, color=color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output

    def convert_cv2QImage(self, cv2_img):
        # OpenCV 이미지를 QImage로 변환하는 메서드
        arr_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        height, width, channels = arr_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(arr_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(q_img)
        return pixmap
