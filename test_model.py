from torchvision import transforms
from PIL import Image
import numpy as np
import torch as tc


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def resize_with_padding_torch(image, target_size=(128, 128)):
    """
    使用 torchvision.transforms 将图像调整为 target_size，同时保持长宽比，并在周围填充黑色。
    """

    # 计算缩放比例
    h, w, c = image.shape
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
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
    print(padded_image.shape)
    padded_image = transforms.Normalize(mean, std)(padded_image).unsqueeze(0)

    return padded_image


def recognize_image(image, model):
    image = resize_with_padding_torch(image, target_size=(128, 128)).to('cuda')
    prediction = model(image)
    predict_ans = tc.argmax(prediction, 1).item()
    return predict_ans, image


def predict_image(model, image_path):
    test_image = Image.open(image_path)
    test_image = np.array(test_image)
    print(test_image.shape)

    prediction, padded_image = recognize_image(test_image, model)
    return prediction
