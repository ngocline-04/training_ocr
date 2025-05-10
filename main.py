# 

import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import re
import io
from ocr.preprocess import recognize_with_tesseract, enhance_cell_image  # Import hàm detect_table_cells từ file riêng

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class OCRProcessor:
    def __init__(self):
        # PaddleOCR hỗ trợ tiếng Việt
        self.ocr = PaddleOCR(lang='vi', use_gpu=False)

        # Cấu hình VietOCR
        self.config = Cfg.load_config_from_file('config/config_after_trainer.yml')
        self.config['weights'] = 'weights/transformerocr.pth'
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cpu'
        self.detector = Predictor(self.config)
    
    def normalize_text(sefl,text):
        text = re.sub(r'\+|t', '7', text)
        text = re.sub(r'S|s', '5', text)
        text = re.sub(r'g|G', '9', text)
        text = re.sub(r'\&|K|k', '8', text)

        if text.isdigit() and len(text) == 2 and int(text) > 10:
            text = f"{text[0]}.{text[1]}"
        return text

    def recognizeText(sefl,image_path):
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
        table_result = []
        for row in rows:
            row_text = []
            for box in row:
                x, y, w, h = box
                cell_img = image[y:y+h, x:x+w]
                result = sefl.ocr.ocr(cell_img, cls=True)
        # text = result[0][0][1][0] if result[0] else ''
                if result and result[0]:
                    text = result[0][0][1][0]
                    
                else:
                    img_pil = enhance_cell_image(cell_img)
                    text = ocr_processor.detector.predict(img_pil)
                # else:
                #      try:
                #           img_pil = enhance_cell_image(cell_img)
                #           text = ocr_processor.detector.predict(img_pil).strip()
                #      except:
                #           text = ""
                # if not text.strip() or text.strip() == ".":
                #     text = recognize_with_tesseract(cell_img)

            # # img_pil.show()
                    #text = recognize_with_tesseract(cell_img)
                # text = re.sub(r'\+|t', '7', text)
                # text = re.sub(r'S|s', '5', text)
                # text = re.sub(r'g|G', '9', text)
                # text = re.
                text = sefl.normalize_text(text)
                print(text,'text======')
                row_text.append(text)
                
            table_result.append(row_text)
        return table_result


ocr_processor = OCRProcessor()


@app.post("/process_image/")
async def process_image(image: UploadFile = File(...)):
    try:
        # Đọc nội dung file ảnh
        contents = await image.read()

        # Chuyển thành file tạm
        image_np = np.frombuffer(contents, np.uint8)
        img_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Lưu vào tạm để sử dụng với VietOCR (nếu cần đường dẫn)
        temp_path = f"temp_{image.filename}"
        cv2.imwrite(temp_path, img_cv2)

        # Gọi hàm nhận diện bảng và text
        result = ocr_processor.recognizeText(temp_path)

        # Xoá file tạm
        os.remove(temp_path)

        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
