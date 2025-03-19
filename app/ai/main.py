from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Resize,
    PILToTensor,
)
from ai_model import YOLOModel, OffsideDetector
from dataset import RoboflowToTorch
from torch.utils.data import DataLoader
from torch import float, argmax
from transforms import YOLOTransformer
from torch.optim import SGD
from torchvision.io import decode_image
from dotenv import load_dotenv
from dataset_loader import DatasetLoader

load_dotenv("./.env")


loader = DatasetLoader()
# player_detection_dataset = loader.roboflow_loader(
#    "nikhil-chapre-xgndf",
#    "detect-players-dgxz0",
#    overwrite=False,
#    location="player-detection/",
# )
offside_dataset = loader.roboflow_loader(
    "football-sioqa",
    "offside-dnedg",
    overwrite=False,
    location="offside/",
    model_format="folder",
)

yolo_model = YOLOModel()


output_channels = 32
kernel_size = 2
img_height = 64
img_width = 64
batch_size = 1

compose = Compose(
    [
        ConvertImageDtype(dtype=float),
        Resize((640, 640)),
        YOLOTransformer(yolo_model),
        PILToTensor(),
        ConvertImageDtype(dtype=float),
        Resize((img_height, img_width)),
    ]
)
train_dataset = RoboflowToTorch("offside", "train", transform=compose)
valid_dataset = RoboflowToTorch("offside", "valid", transform=compose)
#
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
offside_detector = OffsideDetector(
    batch_size=1, img_height=img_height, img_width=img_width
)

optimizer = SGD(offside_detector.get_model_parameters(), lr=0.001, momentum=0.9)


# train models
# yolo_model.train("player-detection/data.yaml", batch=8, epochs=50)
# result_train = offside_detector.train(train_loader, valid_loader, optimizer, epochs=50)


compose.transforms[2].save = True
compose.transforms[2].file_name = "./app/test_2.jpg"
# Predict
image = decode_image("./test_2.jpg", mode="rgb")
result = offside_detector.predict(image, transform=compose)
print(result)

compose.transforms[2].file_name = "./app/offside.jpg"
image = decode_image("./offside.jpg", mode="rgb")
result = offside_detector.predict(image, transform=compose)
print(result)

compose.transforms[2].file_name = "./app/onside.jpg"
image = decode_image("./onside.webp", mode="rgb")
result = offside_detector.predict(image, transform=compose)
print(result)


test = RoboflowToTorch("offside", "test", transform=compose)
loader = DataLoader(test, batch_size=1)
for i, data in enumerate(loader):
    images, labels = data
    compose.transforms[2].file_name = f"./app/{i}.jpg"
    result = offside_detector.predict(images)
    print(result)
    print(labels == argmax(result["predict"].data, dim=1))
