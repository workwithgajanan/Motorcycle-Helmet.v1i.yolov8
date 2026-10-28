import cv2
from ultralytics import YOLO

# Train zalela swatahcha custom model load kara
model = YOLO(r'runs\detect\helmet_model\weights\best.pt')

# Laptop webcam start kara
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Real-time Prediction (With Helmet / Without Helmet)
    results = model(frame, conf=0.4)

    # Bounding boxes draw kara
    annotated_frame = results[0].plot()

    # Window show kara
    cv2.imshow("Custom Helmet Detection Project", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()