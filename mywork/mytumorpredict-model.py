from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(
        data=r'D:\deeplearning\ultralytics-8.3.163\MRI_2D_DATA_USE_3\data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4   # 或更少
    )
# 训练完成后，模型会自动保存为 .pt 文件