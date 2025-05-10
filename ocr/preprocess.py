
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt



def enhance_cell_image(cell_img):
    scale_factor = 2
    resized = cv2.resize(cell_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    img_pil = Image.fromarray(rgb_img)
    return img_pil


import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
def recognize_with_tesseract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Làm nét ảnh nếu cần:
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Chuyển sang ảnh nhị phân để dễ OCR
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Chuyển ảnh về định dạng PIL để pytesseract xử lý
    pil_img = Image.fromarray(binary)

    # Chỉ nhận diện số và dấu chấm
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,'
    text = pytesseract.image_to_string(pil_img, config=config)

    return text.strip()

def group_rows(boxes, y_threshold=15):
    boxes = sorted(boxes, key=lambda b: b[1])  # sort by y
    grouped_rows = []
    current_row = []
    prev_y = -100

    for box in boxes:
        x, y, w, h = box
        if abs(y - prev_y) > y_threshold:
            if current_row:
                grouped_rows.append(current_row)
            current_row = [box]
            prev_y = y
        else:
            current_row.append(box)
    if current_row:
        grouped_rows.append(current_row)
    return grouped_rows

def align_cells_to_columns(rows, num_columns):
    # Tìm vị trí x trung bình của các cột
    column_xs = []
    for row in rows:
        for x, y, w, h in row:
            column_xs.append(x)
    column_xs = sorted(column_xs)
    approx_cols = np.linspace(min(column_xs), max(column_xs), num_columns)

    aligned_rows = []
    for row in rows:
        aligned = [''] * num_columns
        for x, y, w, h in row:
            col_idx = np.argmin([abs(x - cx) for cx in approx_cols])
            aligned[col_idx] = (x, y, w, h)
        aligned_rows.append(aligned)
    return aligned_rows
