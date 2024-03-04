import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# Funkcja do załadowania modelu Faster R-CNN
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    return model


# Funkcja do przetwarzania obrazu
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image)
    return image_tensor, image


# Funkcja do wykrywania i rysowania ramki wokół obiektów na obrazie
def detect_objects(model, image_tensor, image, label_map):
    model.eval()
    with torch.no_grad():
        predictions = model([image_tensor])

    draw = ImageDraw.Draw(image)
    for element in predictions[0]['boxes']:
        draw.rectangle([(element[0], element[1]), (element[2], element[3])], outline="red")

    for i, element in enumerate(predictions[0]['labels']):
        object_name = label_map[element.item()]
        draw.text((predictions[0]['boxes'][i][0], predictions[0]['boxes'][i][1]), f"{object_name}", fill="red")

    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Załadowanie modelu
    model = load_model()

    # Ścieżka do pliku z mapowaniem etykiet
    label_map = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
                 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
                 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
                 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
                 29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
                 35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
                 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
                 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
                 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
                 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
                 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
                 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"}

    # Ścieżka do obrazu, który chcemy przetworzyć
    image_path = 'istockphoto-184276818-612x612.jpg'

    # Przetwarzanie obrazu
    image_tensor, image = preprocess_image(image_path)

    # Wykrywanie obiektów i rysowanie ramki wokół nich
    detect_objects(model, image_tensor, image, label_map)
