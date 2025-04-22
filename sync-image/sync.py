import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from faker import Faker
import os

# Khởi tạo Faker để tạo tên sinh viên ngẫu nhiên
fake = Faker('vi_VN')

# Thư mục lưu ảnh
output_dir = "dataset/dataline/synthetic_images"
os.makedirs(output_dir, exist_ok=True)

handwritten_fonts = [
    "font/handwritting.otf",
    "font/handwritting_v2.ttf",
    "font/handwritting_v3.ttf",
    "font/handwritting_v4.ttf"
]
typed_font = "font/arial.ttf"

# Hàm tạo ngày sinh ngẫu nhiên
def random_birthday():
    return fake.date_of_birth(minimum_age=18, maximum_age=25).strftime("%d/%m/%Y")

# Hàm tạo điểm ngẫu nhiên
def random_score(min_val=5, max_val=10):
    return round(random.uniform(min_val, max_val), 1)

def generate_vietnamese_name():
    full_name = fake.name()
    parts = full_name.split()

    if len(parts) >= 2:
        ten = " ".join(parts[:-1])
        ho_tendem = parts[-1]
    else:
        ho_tendem = full_name
        ten = full_name  # fallback nếu không tách được

    return ho_tendem, ten

def create_score_sheet(idx, handwritten_font):
    img = np.ones((350, 1000, 3), dtype=np.uint8) * 255  # Tạo ảnh nền trắng
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Chọn font chữ viết tay theo idx
    handwritten_font = handwritten_fonts[idx % 4]
    font_handwritten = ImageFont.truetype(handwritten_font, 30)
    font_typed = ImageFont.truetype(typed_font, 25)

    # Sinh dữ liệu giả
    ma_sv = f"{random.randint(1, 99)}{random.choice(['A', 'B', 'C'])}{random.randint(100000, 999999)}"
    # ho_tendem = fake.last_name()
    # ten_sv = fake.first_name()
    ho_tendem, ten_sv = generate_vietnamese_name()
    lop = f"K{random.randint(1, 99)}{random.choice(['A', 'B', 'C'])}{random.randint(100000, 999999)}"
    ngay_sinh = random_birthday()
    diem_cc = random_score(0, 10)
    diem_kt1 = random_score(0, 10)
    diem_kt2 = random_score(0, 10)
    so_to = random.randint(1, 5)
    ky_nop = "✓" if random.random() > 0.1 else "X"
    diem_thi = random_score(0, 10)
    ghi_chu = random.choice(["", "Bảo lưu", "Vắng thi", ""])
    ghi_chu = ghi_chu if ghi_chu else "None"

    # Vẽ thông tin lên ảnh
    line_gap = 50
    # draw.text((20, 20 + 0 * line_gap), f"SBD: {sbd}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 1 * line_gap), f"Mã SV: {ma_sv}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 2 * line_gap), f"Họ và tên lót : {ho_tendem}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 2 * line_gap), f"Tên : {ten_sv}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 3 * line_gap), f"Lớp : {lop}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 3 * line_gap), f"Ngày Sinh: {ngay_sinh}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 4 * line_gap), f"Điểm CC: {diem_cc}", font=font_typed, fill=(0, 0, 0))
    # draw.text((320, 20 + 4 * line_gap), f"KT1: {diem_kt1}", font=font_typed, fill=(0, 0, 0))
    # draw.text((420, 20 + 4 * line_gap), f"KT2: {diem_kt2}", font=font_typed, fill=(0, 0, 0))
    # draw.text((20, 20 + 5 * line_gap), f"Số tờ: {so_to}", font=font_typed, fill=(0, 0, 0))
    # draw.text((180, 20 + 5 * line_gap), f"Ký nộp: {ky_nop}", font=font_typed, fill=(0, 0, 0))
    # # draw.text((320, 20 + 5 * line_gap), f"Điểm thi: {diem_thi}", font=font_handwritten, fill=(0, 0, 0))
    # draw.text((320, 20 + 5 * line_gap), "Điểm thi:", font=font_typed, fill=(0, 0, 0))
    # # Ghi điểm số bằng chữ viết tay, đặt ngay sau "Điểm thi:"
    # # Ước lượng chiều rộng của đoạn "Điểm thi:" để căn chỉnh vị trí ghi điểm
    # # *** Changed line: Use font_typed.getsize instead of draw.textsize ***
    # bbox = font_typed.getbbox("Điểm thi:")
    # text_width = bbox[2] - bbox[0]
    # draw.text((320 + text_width + 10, 20 + 5 * line_gap), str(diem_thi), font=font_handwritten, fill=(0, 0, 0))
    # # ... (rest of your create_score_sheet function)

    # draw.text((500, 20 + 5 * line_gap), f"Ghi chú: {ghi_chu}", font=font_typed, fill=(0, 0, 0))
    column_gap = 330  # hoặc 1000 / 3 nếu bạn có 3 cột
    # Dòng 1: SBD - Mã SV - Họ tên
    draw.text((20 + 0 * column_gap, 20), f"Mã SV: {ma_sv}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 1 * column_gap, 20), f"Họ và tên lót: {ho_tendem}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 2 * column_gap, 20), f"Tên: {ten_sv}", font=font_typed, fill=(0, 0, 0))
    # Dòng 2: Lớp - Ngày sinh - Số tờ
    draw.text((20 + 0 * column_gap, 20 + 1 * line_gap), f"Lớp: {lop}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 1 * column_gap, 20 + 1 * line_gap), f"Ngày sinh: {ngay_sinh}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 2 * column_gap, 20 + 1 * line_gap), f"Số tờ: {so_to}", font=font_typed, fill=(0, 0, 0))
    
    # Dòng 3: Điểm CC - KT1 - KT2
    draw.text((20 + 0 * column_gap, 20 + 2 * line_gap), f"Điểm CC: {diem_cc}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 1 * column_gap, 20 + 2 * line_gap), f"KT1: {diem_kt1}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 2 * column_gap, 20 + 2 * line_gap), f"KT2: {diem_kt2}", font=font_typed, fill=(0, 0, 0))
    
    # Dòng 4: Ký nộp - Điểm thi - Ghi chú
    draw.text((20 + 0 * column_gap, 20 + 3 * line_gap), f"Ký nộp: {ky_nop}", font=font_typed, fill=(0, 0, 0))
    draw.text((20 + 1 * column_gap, 20 + 3 * line_gap), "Điểm thi:", font=font_typed, fill=(0, 0, 0))
    text_width = font_typed.getbbox("Điểm thi:")[2]
    draw.text((20 + 1 * column_gap + text_width + 10, 20 + 3 * line_gap), str(diem_thi), font=font_handwritten, fill=(0, 0, 0))
    draw.text((20 + 2 * column_gap, 20 + 3 * line_gap), f"Ghi chú: {ghi_chu}", font=font_typed, fill=(0, 0, 0))


    img_path = os.path.join(output_dir, f"score_{idx}.jpg")
    pil_img.save(img_path)
    print(f"Đã lưu ảnh vào: {img_path}")

    # Xuất nhãn đầy đủ theo format yêu cầu
    label = f"MãSV:{ma_sv}, Họ và tên lót:{ho_tendem}, Tên:{ten_sv}, NgàySinh:{ngay_sinh}, ĐiểmCC:{diem_cc}, KT1:{diem_kt1}, KT2:{diem_kt2}, SốTờ:{so_to}, KýNộp:{ky_nop}, ĐiểmThi:{diem_thi}, GhiChú:{ghi_chu}"
    # Combine all data into a single label string

    # Write image path and label to the file with a tab separator
    row = f"synthetic_images/score_{idx}.jpg\t{label}"

    return row

# # Chia dữ liệu: 80% train, 20% test
num_images = 4000
train_ratio = 0.8  # 80% dữ liệu dùng để train
num_train = int(num_images * train_ratio)  # 2400 ảnh cho train
num_test = num_images - num_train  # 600 ảnh cho test

train_rows = []
test_rows = []

for i in range(num_images):
    # Provide a handwritten font path here, or modify create_score_sheet to not require it
    # For example, you could use the first font in your handwritten_fonts list:
    handwritten_font_path = handwritten_fonts[0]
    row = create_score_sheet(i, handwritten_font_path)  # Pass handwritten_font_path to create_score_sheet

    # Chia tập dữ liệu
    if i < num_train:
        train_rows.append(row)
    else:
        test_rows.append(row)

# Ghi ra file
with open("dataset/dataline/train_line_annotation.txt", "w", encoding="utf-8") as train_f, \
     open("dataset/dataline/test_line_annotation.txt", "w", encoding="utf-8") as test_f:

    train_f.write("\n".join(train_rows))
    test_f.write("\n".join(test_rows))

print("Hoàn thành tạo ảnh và ghi nhãn.")