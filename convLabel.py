import os
from PIL import Image


def yolo_to_bbox(yolo_file, image_width, image_height):
    with open(yolo_file, 'r') as file:
        lines = file.readlines()

    bboxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        bboxes.append((int(class_id), x_min, y_min, x_max, y_max))

    return bboxes


def process_dataset(images_dir, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            # İlgili görüntü dosyasını bul
            image_file = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))
            if not os.path.exists(image_file):
                image_file = image_file.replace(".jpg", ".png")  # PNG için kontrol
                if not os.path.exists(image_file):
                    print(f"Image not found for label: {label_file}")
                    continue

            # Görüntü boyutlarını al
            with Image.open(image_file) as img:
                image_width, image_height = img.size

            # YOLO etiketlerini dönüştür
            yolo_file_path = os.path.join(labels_dir, label_file)
            bboxes = yolo_to_bbox(yolo_file_path, image_width, image_height)

            # Dönüşümü kaydet
            output_file_path = os.path.join(output_dir, label_file)
            with open(output_file_path, 'w') as f:
                for bbox in bboxes:
                    f.write(" ".join(map(str, bbox)) + "\n")

    print(f"Dönüştürme tamamlandı: {output_dir}")


# Ana dizin
dataset_path = "/Users/altun/Desktop/veriSeti"

# Eğitim veri setini dönüştür
process_dataset(
    images_dir=os.path.join(dataset_path, "train/images"),
    labels_dir=os.path.join(dataset_path, "train/labels"),
    output_dir=os.path.join(dataset_path, "train/converted_labels")
)

# Doğrulama veri setini dönüştür
process_dataset(
    images_dir=os.path.join(dataset_path, "val/images"),
    labels_dir=os.path.join(dataset_path, "val/labels"),
    output_dir=os.path.join(dataset_path, "val/converted_labels")
)

# Test veri setini dönüştür
process_dataset(
    images_dir=os.path.join(dataset_path, "test/images"),
    labels_dir=os.path.join(dataset_path, "test/labels"),
    output_dir=os.path.join(dataset_path, "test/converted_labels")
)