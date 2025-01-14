import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")  # veya kendi eğittiğiniz modelin yolunu buraya yazabilirsiniz

# Web kamerasını başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # Modeli kare üzerinde çalıştır
    results = model(frame)

    # Tahmin sonuçlarını görselleştir
    annotated_frame = results[0].plot()


    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()