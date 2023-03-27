import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import requests
import io


from itertools import permutations
colors = [*permutations((255, 0, 0)), *permutations((255, 128, 0)), *permutations((255, 128, 128))]

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def get_model(device):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.eval()
    return model.to(device)


def object_detection(img, model, device, threshold=0.5):
    # Transform the image
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor.to(device))

    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    img_np = T.ToPILImage()(img_tensor.squeeze(0))
    draw = ImageDraw.Draw(img_np)
    font = ImageFont.truetype("RobotoCondensed-Bold.ttf", 30)
    # font = ImageFont.load_default()

    for i in range(boxes.size(0)):
        if scores[i] >= threshold:
            box = boxes[i].tolist()
            label = COCO_CLASSES[labels[i].item()]
            color = colors[labels[i].item() % len(colors)]

            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 5, box[1]), label, fill=color, font=font)
    return img_np


def img_from_url(url):
    try:
        data = requests.get(url).content
    except requests.exceptions.InvalidSchema:
        return "wrong URL schema"
    except requests.exceptions.ConnectionError:
        return 'server not found or connection fault'
    try:
        img = Image.open(io.BytesIO(data)).convert('RGB')
    except UnidentifiedImageError:
        return 'not an image URL'
    return img
