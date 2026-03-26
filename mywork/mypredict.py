from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source=r"D:\deeplearning\ultralytics-8.3.163\ultralytics\assets",
    save=True,
    show=False,
    conf=0.0,
    iou=1.0,
    max_det=9999,
)
