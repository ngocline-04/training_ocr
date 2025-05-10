# import sys
# import os

# # Thêm đường dẫn cha chứa thư mục craft_structure vào PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import cv2
# import torch
# import time
# from craft_structure.detection import detect, get_detector
# import matplotlib.pyplot as plt
# from preprocess import detect_table_cells

# device = torch.device("cpu")
# craft = get_detector("./weights/craft_mlt_25k.pth", device)

# def craftTextDetection(cells, table_image):
#  final_horizontal_list = []
#  for cell in cells:
#     cell_x_min, cell_y_min, cell_x_max, cell_y_max = cell
#     cell_image = table_image[cell_y_min:cell_y_max, cell_x_min:cell_x_max]

#     # horizontal_list, free_list = detect(craft, cell_image, device=device)
#     if len(cell_image) == 0 or cell_image.shape[0] < 5 or cell_image.shape[1] < 5:
#      continue  # Tránh gửi ảnh quá nhỏ

#     try:
#        horizontal_list, free_list = detect(craft, cell_image, device=device)
#        if not horizontal_list:
#           continue  # Không có box nào được phát hiện
#     except Exception as e:
#        print(f"Lỗi phát hiện ở ô bảng: {e}")
#        continue

#     for box in horizontal_list:
#         x_min = cell_x_min + box[0]
#         x_max = cell_x_min + box[1]
#         y_min = cell_y_min + box[2]
#         y_max = cell_y_min + box[3]
#         cv2.rectangle(table_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         final_horizontal_list.append([x_min, x_max, y_min, y_max])

    

# if __name__ == "__main__": 
#        # Đọc ảnh và phát hiện ô bảng
#     image_path = "img/6.JPG"
#     output_img, table_cells = detect_table_cells(image_path)

#     # Gọi CRAFT để nhận diện chữ trong từng ô
#     craftTextDetection(table_cells, output_img)

#     # Hiển thị ảnh đã vẽ vùng chữ bằng matplotlib
#     output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(12, 10))
#     plt.imshow(output_img_rgb)
#     plt.title("CRAFT text detection inside table cells")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import matplotlib.pyplot as plt

# Thêm đường dẫn cha chứa thư mục preprocess vào PYTHONPATH


import numpy as np

from preprocess import detect_table_cells

from paddleocr import PaddleOCR


# Khởi tạo PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='vi')  # Hỗ trợ tiếng Việt

def recognize_cells(table_image, cells):
    results = []
    for idx, cell in enumerate(cells):
        x_min, y_min, x_max, y_max = cell
        cell_img = table_image[y_min:y_max, x_min:x_max]

        # Nhận diện chữ trong từng ô bảng
        result = ocr.ocr(cell_img, cls=True)

        # Lưu kết quả dạng text cho từng ô
        cell_text = ''
        if result:
            for line in result:
                if line:  # tránh None
                    for word in line:
                        cell_text += word[1][0] + ' '
        
        results.append({
            "cell_index": idx,
            "bbox": cell,
            "text": cell_text.strip()
        })

        # Vẽ bounding box và text lên ảnh để visualize
        cv2.rectangle(table_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(table_image, str(idx), (x_min + 3, y_min + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return results, table_image

# def preprocess_and_deskew(img):
#     # Resize nhỏ lại nếu quá to
#     h, w = img.shape[:2]
#     if max(h, w) > 1000:
#         scale = 1000.0 / max(h, w)
#         img = cv2.resize(img, (int(w * scale), int(h * scale)))

#     # Làm sáng + tăng tương phản
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     enhanced = cv2.merge((cl, a, b))
#     img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

#     # Làm mịn nhẹ rồi làm nét
#     img_blur = cv2.GaussianBlur(img, (3, 3), 0)
#     img_sharp = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0)

#     # Chuyển xám + nhị phân để chống nghiêng
#     gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Tính góc nghiêng dựa trên hình chữ nhật bao quanh nội dung
#     coords = np.column_stack(np.where(binary > 0))
#     if len(coords) > 0:
#         rect = cv2.minAreaRect(coords)
#         angle = rect[-1]
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle

#         (h, w) = img.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         deskewed = cv2.warpAffine(img_sharp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#         return deskewed
#     else:
#         return img_sharp

def preprocess_and_deskew(img, output_dir="processed_images"):
    # Kiểm tra và tạo thư mục lưu ảnh nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Resize nhỏ lại nếu quá to
    h, w = img.shape[:2]
    if max(h, w) > 1000:
        scale = 1000.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Làm sáng + tăng tương phản
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Làm mịn nhẹ rồi làm nét
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_sharp = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0)

    # Chuyển xám + nhị phân để chống nghiêng
    gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tính góc nghiêng dựa trên hình chữ nhật bao quanh nội dung
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Chỉ xoay ảnh nếu góc nghiêng lớn hơn một ngưỡng nhất định (vd: -10 < angle < 10)
        if angle < -10 or angle > 10:
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(img_sharp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed = img_sharp
    else:
        deskewed = img_sharp

    # Lưu ảnh đã xử lý vào thư mục
    output_path = os.path.join(output_dir, "processed_image.jpg")
    cv2.imwrite(output_path, deskewed)

    return output_path
    
if __name__ == "__main__":
    image_path = "img/6.JPG"
    img = cv2.imread(image_path)
    processed_img = preprocess_and_deskew(img)
    
    # Tiến hành phát hiện các ô bảng từ ảnh đã xử lý
    output_img, table_cells = detect_table_cells(processed_img)
    recognized_results, vis_img = recognize_cells(output_img, table_cells)
    # Hiển thị ảnh kết quả có các bounding box từ detect_table_cells
    #output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("PaddleOCR nhận diện văn bản trong từng ô")
    plt.axis("off")
    plt.show()

    for item in recognized_results:
        print(f"[Cell {item['cell_index']}] {item['text']}")

    
    
    
    
    
    # # In kết quả dạng hàng-cột
    # for item in recognized_results:
    #     print(f"[Cell {item['cell_index']}] {item['text']}")
    # image_path = "img/6.JPG"
    # table_img, table_cells = detect_table_cells(image_path)

    # print(f"Đã phát hiện {len(table_cells)} ô bảng.")
    # recognized_results, vis_img = recognize_cells(table_img, table_cells)

    # # Hiển thị ảnh kết quả có box + text
    # plt.figure(figsize=(12, 10))
    # plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    # plt.title("PaddleOCR nhận diện văn bản trong từng ô")
    # plt.axis("off")
    # plt.show()

    # # In kết quả dạng hàng-cột
    # for item in recognized_results:
    #     print(f"[Cell {item['cell_index']}] {item['text']}")
