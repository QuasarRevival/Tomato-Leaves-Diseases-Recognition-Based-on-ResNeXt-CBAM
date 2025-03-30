from ultralytics import YOLO
import torchvision
import torch

train_path = '../datasets/DetectTrain/data.yaml'

print(torch.cuda.is_available())
print(torchvision.ops.box_convert)

'''
# 测试 NMS（应在 GPU 运行）
boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9], dtype=torch.float32).cuda()
torchvision.ops.nms(boxes, scores, 0.5)  # 应不报错
'''


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    yolo_model = YOLO("yolov8n.pt").to('cuda')
    yolo_model.train(data=train_path, epochs=100, imgsz=224, device=0)
