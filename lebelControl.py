import os
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def draw_bboxes(image_path, bboxes):
    """
    Görüntü üzerinde bounding box'ları çiz.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max = bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.text((x_min, y_min - 10), f"Class: {class_id}", fill="red")

    return image


def load_bboxes(label_file):
    """
    Etiket dosyasını yükle ve bounding box bilgilerini al.
    """
    with open(label_file, 'r') as file:
        lines = file.readlines()
    bboxes = [list(map(int, line.strip().split())) for line in lines]
    return bboxes


def visualize_dataset(images_dir, labels_dir, num_rows=5, num_cols=3):
    """
    Rastgele seçilen görselleri ve etiket kutularını yan yana görselleştir.
    """
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()

    for ax in axes:
        if image_files:
            image_file = image_files.pop()
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.join(labels_dir, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

            if os.path.exists(label_file):
                bboxes = load_bboxes(label_file)
                image_with_bboxes = draw_bboxes(image_path, bboxes)
                ax.imshow(image_with_bboxes)
                ax.axis("off")
            else:
                ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Ana dizin
dataset_path = "/Users/altun/Desktop/veriSeti"
train_images = os.path.join(dataset_path, "train/images")
train_labels = os.path.join(dataset_path, "train/converted_labels")  # Piksel tabanlı etiketler

# Görselleştir
visualize_dataset(train_images, train_labels, num_rows=5, num_cols=3)