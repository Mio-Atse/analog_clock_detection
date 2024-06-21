import os
import torch
from PIL import Image

# Model yükleniyor
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Klasör yolları
image_dir = 'images'
fixed_image_dir = 'imaged_cropped_fixed'  # Yeni oluşturulan klasör adı

# Kırpılan ve boyutlandırılan resimlerin kaydedileceği klasör yoksa oluşturuluyor
if not os.path.exists(fixed_image_dir):
    os.makedirs(fixed_image_dir)

# 'images' klasöründeki tüm resim dosyalarını al
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    # Resmi aç
    with Image.open(image_path) as img:
        results = model(img)
        detections = results.pandas().xyxy[0]

        if not detections.empty:
            # İlk tespiti alın
            xmin, ymin, xmax, ymax = int(detections.iloc[0]['xmin']), int(detections.iloc[0]['ymin']), int(detections.iloc[0]['xmax']), int(detections.iloc[0]['ymax'])

            # Nesne kırpılıyor
            cropped_image = img.crop((xmin, ymin, xmax, ymax))

            # Boyutlandırma ve kaydetme işlemi
            cropped_image.thumbnail((800, 800))
            fixed_image_path = os.path.join(fixed_image_dir, image_file)
            cropped_image.save(fixed_image_path)
            print(f"Resized and saved: {fixed_image_path}")
        else:
            # Nesne tespit edilemezse orijinal resmi boyutlandır ve kaydet
            img.thumbnail((800, 800))
            fixed_image_path = os.path.join(fixed_image_dir, image_file)
            img.save(fixed_image_path)
            print(f"Resized and saved: {fixed_image_path}")

print("Boyutlandırma ve kaydetme işlemi tamamlandı.")
