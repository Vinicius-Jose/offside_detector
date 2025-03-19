from dotenv import load_dotenv

load_dotenv("./.env")

from .dataset_loader import DatasetLoader
from .dataset import RoboflowToTorch
from .ai_model import YOLOModel, OffsideDetector
from .transforms import YOLOTransformer

from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Resize,
    PILToTensor,
)

from torchvision.io import decode_image
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
import os
from PIL import Image
import base64
from io import BytesIO


class AIFacade:

    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size: int = batch_size
        self.yolo_model: YOLOModel = YOLOModel()
        img_height: int = 64
        img_width: int = 64
        self.__optimizer: Optimizer = None
        self.offside_detector = OffsideDetector(
            batch_size=self.batch_size, img_height=img_height, img_width=img_width
        )
        self.predict_compose = Compose(
            [
                ConvertImageDtype(dtype=torch.float),
                Resize((640, 640)),
                YOLOTransformer(self.yolo_model),
            ]
        )
        self.compose = Compose(
            [
                self.predict_compose,
                PILToTensor(),
                ConvertImageDtype(dtype=torch.float),
                Resize((img_height, img_width)),
            ]
        )

    def predict(
        self,
        image_path: str,
    ) -> dict:
        height, width = Image.open(image_path).size
        image = decode_image(image_path, mode="rgb")
        result = self.offside_detector.predict(image, transform=self.compose)
        yolo_image: Image = self.predict_compose(image)
        result["image_predicted_base64"] = self.__cast_pill_image_to_base64(
            yolo_image, height, width
        )
        result["image_predicted_pill"] = yolo_image
        return result

    def train(self, epochs: int = 50):
        self.get_optimizer()
        datasets = self.__load_datasets_roboflow()
        yaml_path = os.path.join(datasets[2], "data.yaml")
        self.yolo_model.train(yaml_path, batch=8, epochs=epochs)
        result_train = self.offside_detector.train(
            datasets[0], datasets[1], self.__optimizer, epochs=epochs
        )
        return result_train

    def __load_datasets_roboflow(
        self,
    ) -> list[dict]:
        loader = DatasetLoader()
        player_detection_dataset = loader.roboflow_loader(
            "nikhil-chapre-xgndf",
            "detect-players-dgxz0",
            overwrite=False,
            location="player-detection/",
        )

        offside_dataset = loader.roboflow_loader(
            "football-sioqa",
            "offside-dnedg",
            overwrite=False,
            location="offside/",
            model_format="folder",
        )
        train_dataset = RoboflowToTorch("offside", "train", transform=self.compose)
        valid_dataset = RoboflowToTorch("offside", "valid", transform=self.compose)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size)
        return (train_loader, valid_loader, "player-detection/")

    def get_optimizer(self) -> Optimizer:
        if not self.__optimizer:
            self.set_optimizer(
                SGD(
                    self.offside_detector.get_model_parameters(), lr=0.001, momentum=0.9
                )
            )
        return self.__optimizer

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.__optimizer = optimizer

    def __cast_pill_image_to_base64(self, image: Image, height: int, width: int) -> str:
        buffer = BytesIO()
        image = image.resize((height, width))
        image.save(buffer, format="jpeg")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64
