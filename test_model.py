from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch as tc

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


image_path = './datasets/PlantVillageTomatoLeavesDataset/test/Powdery_mildew/pm66_lower.jpg'
yolo_model_path = 'runs/detect/train2/weights/best.pt'


def resize_with_padding_torch(image, crop_size=(224, 224)):
    """
    使用 torchvision.transforms 将图像调整为 target_size，同时保持长宽比，并在周围填充黑色。
    """

    # 计算缩放比例
    target_size = (256, 256)
    h, w, c = image.shape
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # plt.imshow(image)
    # plt.show()
    image = transforms.ToTensor()(image)
    image.permute([2, 0, 1])
    print(image.shape)

    # 缩放图像
    resize_transform = transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
    resized_image = resize_transform(image)

    # 计算填充大小
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    # 填充图像
    pad_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
    padded_image = pad_transform(resized_image)
    crop_image = transforms.CenterCrop(crop_size)(padded_image)
    plt.imshow(crop_image.permute([1, 2, 0]).to('cpu').numpy())
    plt.show()
    # print(padded_image.shape)
    crop_image = transforms.Normalize(mean, std)(padded_image).unsqueeze(0)

    return crop_image


def open_image(image_path):
    test_image = Image.open(image_path)
    test_image = np.array(test_image)
    return test_image


def recognize_image(image, model):
    image = resize_with_padding_torch(image, crop_size=(224, 224)).to('cuda')
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean, std)
    ])
    image = transform(image).unsqueeze(0).to('cuda')
    plt.imshow(image.squeeze().permute([1, 2, 0]).to('cpu').numpy())
    plt.show()
    '''
    prediction = model(image)
    print(prediction)
    predict_ans = tc.argmax(prediction, 1).item()
    return predict_ans, image


def predict_image(model, test_image):
    prediction, padded_image = recognize_image(test_image, model)
    return prediction


def load_yolo_model(yolo_path):
    model = YOLO(yolo_path)
    return model


def detect_object(model, img_path):
    results = model.predict(img_path, conf=0.4)
    return results


def cut_single_image(img_path, yolo_path):
    model = load_yolo_model(yolo_path)
    results = detect_object(model, img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > 0:
        results[0].show()
        x1, y1, x2, y2 = boxes[0].astype(int)
        roi = open_image(img_path)[y1:y2, x1:x2]
        # cv2.imwrite("cropped_leaf.jpg", roi)
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return roi
    return None


def predict_from_extended(img_path, yolo_path, model):
    roi = cut_single_image(img_path, yolo_path)
    if roi is None:
        return None
    result = predict_image(model, roi)
    return result


# cut_single_image(image_path, yolo_model_path)
