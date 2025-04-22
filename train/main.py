from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name':'hw', # tên dataset do bạn tự đặt
    'data_root':'dataset/dataline', # thư mục chứa dữ liệu bao gồm ảnh và nhãn
    'train_annotation':'train_line_annotation.txt', # ảnh và nhãn tập train
    'valid_annotation':'test_line_annotation.txt' # ảnh và nhãn tập test
}
params = {
         'print_every':200, # hiển thị loss mỗi 200 iteration 
         'valid_every':10000, # đánh giá độ chính xác mô hình mỗi 10000 iteraction
          'iters':20000, # Huấn luyện 20000 lần
          'export':'./weights/transformerocr.pth', # lưu model được huấn luyện tại này
          'metrics': 4000 # sử dụng 4000 ảnh của tập test để đánh giá mô hình
         }


config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cpu'
config['vocab'] += '✓'
config['dataloader']['num_workers'] = 0
# huấn luyện mô hình từ pretrained model của mình sẽ nhanh hội tụ và cho kết quả tốt hơn khi bạn chỉ có bộ dataset nhỏ
# để sử dụng custom augmentation, các bạn có thể sử dụng Trainer(config, pretrained=True, augmentor=MyAugmentor()) theo ví dụ trên.
trainer = Trainer(config, pretrained=True)

# sử dụng lệnh này để visualize tập train, bao gồm cả augmentation
trainer.visualize_dataset()
# # bắt đầu huấn luyện
trainer.train()
print("✅ Quá trình huấn luyện hoàn tất!")

# visualize kết quả dự đoán của mô hình
trainer.visualize_prediction()

# huấn luyện xong thì nhớ lưu lại config để dùng cho Predictor
trainer.config.save('config.yml')