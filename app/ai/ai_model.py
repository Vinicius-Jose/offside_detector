from numpy import ndarray
from torch import Tensor, argmax, no_grad, save, load
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics import settings
from torch.utils.data import DataLoader
from torch.nn import (
    Sequential,
    Linear,
    Conv2d,
    MaxPool2d,
    Softmax,
    Flatten,
    CrossEntropyLoss,
    ReLU,
)
from torch.optim import Optimizer
import os

from torchvision.transforms import Compose


DEFAULT_MODEL_FOLDER = os.getenv("DEFAULT_AI_MODEL_FOLDER")


class YOLOModel:
    def __init__(self, saved_model: str = "yolo_model.pt") -> None:
        super(YOLOModel).__init__()
        self.default_dataset_folder: str = os.getenv("DEFAULT_DATASET_FOLDER")
        settings.update({"datasets_dir": self.default_dataset_folder})
        self.saved_model_path: str = os.path.join(DEFAULT_MODEL_FOLDER, saved_model)
        self.__model: YOLO = self.__load_model()

    def train(
        self,
        dataset: str,
        epochs: int = 50,
        batch: int = 1,
        workers: int = 2,
        imgsz: int = 640,
    ) -> dict:
        dataset = self.default_dataset_folder + dataset
        results = self.__model.train(
            data=dataset, epochs=epochs, batch=batch, workers=workers, imgsz=imgsz
        )
        self.__model.save(self.saved_model_path)
        return results

    def predict(
        self,
        image: str | int | list | tuple | ndarray | Tensor,
        conf: float = 0.4,
        batch: int = 1,
        save: bool = False,
        imgsz: int = 640,
    ) -> list[Results]:
        results = self.__model.predict(
            image, save=save, conf=conf, batch=batch, imgsz=imgsz
        )
        return results

    def __load_model(self) -> YOLO:
        file_path = self.saved_model_path
        if not os.path.isfile(file_path):
            file_path = os.path.join(DEFAULT_MODEL_FOLDER, "yolo11n.pt")
        return YOLO(file_path, task="detect")


class OffsideDetector:
    def __init__(
        self,
        img_height: int = 640,
        img_width: int = 640,
        batch_size: int = 1,
        saved_model: str = "offside_detector.pt",
        classes: list[str] = ["offside", "onside"],
    ) -> None:
        self.classes = classes
        super(OffsideDetector, self).__init__()
        self.saved_model_path = os.path.join(DEFAULT_MODEL_FOLDER, saved_model)
        output_channels: int = 32
        kernel_size: int = 2
        linnear_in_features: int = (
            output_channels
            * (img_height - kernel_size)
            * (img_width - kernel_size)
            * batch_size
        )
        self.__model: Sequential = Sequential(
            Conv2d(3, 16, (kernel_size, kernel_size)),
            Conv2d(16, output_channels, (kernel_size, kernel_size)),
            MaxPool2d((1, 1)),
            Flatten(),
            Linear(linnear_in_features, 120),
            ReLU(),
            Linear(120, 60),
            ReLU(),
            Linear(60, 2),
            Softmax(dim=1),
        )
        self.__load_model()

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: Optimizer,
        epochs: int = 50,
    ) -> dict:

        cost_list = []
        accuracy_list = []
        cost = 0
        criterion = CrossEntropyLoss()
        val_size = len(validation_loader.dataset)
        for epoch in range(epochs):
            self.__model.train()
            cost = self.__execute_training(train_loader, optimizer, criterion)
            cost_list.append(cost)
            if epoch % 10 == 0:
                accuracy = self.__validate(validation_loader, val_size)
                accuracy_list.append(accuracy)
            if cost == min(cost_list):
                save(self.__model.state_dict(), self.saved_model_path)
        return {"cost_list": cost_list, "accuracy_list": accuracy_list}

    def __execute_training(self, train_loader, optimizer, criterion):
        cost = 0
        for _, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            predict = self.__model(images)
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
            cost += loss.item()
        return cost

    def __validate(self, validation_loader: DataLoader, val_size: int):
        correct = 0
        self.__model.eval()
        for _, data in enumerate(validation_loader):
            images, labels = data
            predict = self.__model(images)
            yhat = argmax(predict.data, dim=1)
            correct += (yhat == labels).sum().item()

        accuracy = correct / val_size
        return accuracy

    def predict(
        self,
        image: Tensor,
        transform: Compose | object | None = None,
    ):
        self.__model.eval()
        with no_grad():
            if transform:
                image = transform(image)
                image = image.unsqueeze(0)
            predict = self.__model(image)
            yhat = argmax(predict.data, dim=1)
            return {
                "result": self.classes[yhat.item()],
                "predict": predict,
                "offside": predict[0][0].item(),
                "onside": predict[0][1].item(),
            }

    def get_model_parameters(self) -> dict:
        return self.__model.parameters()

    def __load_model(self) -> None:
        file_path = self.saved_model_path
        if os.path.isfile(file_path):
            self.__model.load_state_dict(load(self.saved_model_path, weights_only=True))
