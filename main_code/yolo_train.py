from ultralytics import YOLO
model = YOLO("./weights/yolov8l.pt")
#box,default 7.5,Weight of the box loss component in the loss function,
# influencing how much emphasis is placed on accurately predicting bounding box coordinates.

if __name__ == "__main__":
    results = model.train(data="../images/climbing.yaml", epochs=20, batch=16, box=7.5)