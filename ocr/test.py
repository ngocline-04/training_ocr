import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from ocr.preprocess import enhance_cell_image, recognize_with_tesseract
vietocr_config = Cfg.load_config_from_name('vgg_transformer')
vietocr_config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
vietocr_config['device'] = 'cpu'  # hoặc 'cuda' nếu bạn dùng GPU
vietocr_model = Predictor(vietocr_config)
# Load ảnh
image_path = 'img/test.png'  # <-- sửa đường dẫn nếu cần
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm nét + nhị phân
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

# Tìm hàng
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Tìm cột
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

# Hợp lại để có khung bảng
table_mask = cv2.add(detect_horizontal, detect_vertical)

# Tìm contours của ô
contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Lọc những contour có kích thước phù hợp (ô bảng)
boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 30 and h > 20:  # Tùy chỉnh theo ảnh
        boxes.append((x, y, w, h))

# Sắp xếp ô theo tọa độ từ trên xuống dưới, trái qua phải
boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

# Nhóm các ô theo hàng
rows = []
current_row = []
prev_y = -100
for box in boxes:
    x, y, w, h = box
    if abs(y - prev_y) > 20:
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
        current_row = [box]
        prev_y = y
    else:
        current_row.append(box)
if current_row:
    rows.append(sorted(current_row, key=lambda b: b[0]))

# OCR từng ô
ocr = PaddleOCR(use_angle_cls=True, lang='vi')

table_result = []
for row in rows:
    row_text = []
    for box in row:
        x, y, w, h = box
        cell_img = image[y:y+h, x:x+w]
        result = ocr.ocr(cell_img, cls=True)
        # text = result[0][0][1][0] if result[0] else ''
        if result and result[0]:
            text = result[0][0][1][0]
        else:
            # img_pil = enhance_cell_image(cell_img)
            # # img_pil.show()
            text = recognize_with_tesseract(cell_img)

        row_text.append(text)
    table_result.append(row_text)
print(table_result, 'result')
# In bảng kết quả
for row in table_result:
    print('\t'.join(row))
