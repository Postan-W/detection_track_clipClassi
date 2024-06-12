from ultralytics import YOLO
import cv2
model = YOLO("./weights/yolov8m20240606.pt")

results = model(source="./images/positive1_10.jpg",classes=[1])
for result in results:
    print(result.boxes.data.tolist())
    frame = result.orig_img
    cv2.imshow('Image with Detections', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    frame = result.plot()
    cv2.imshow('Image with Detections', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

