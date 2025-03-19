from torch.utils.data import Dataset
import os
from torch import tensor
from torchvision.io import decode_image
from torchvision.transforms import Compose


class RoboflowToTorch(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        type_dataset: str = None,
        transform: Compose | object | None = None,
        target_transform=None,
        classes: list[str] = ["offside", "onside"],
    ):
        self.default_dataset_folder = os.getenv("DEFAULT_DATASET_FOLDER")
        self.img_dir = os.path.join(self.default_dataset_folder, dataset_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.img_labels = self.__load_images(type_dataset)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = decode_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __load_images(
        self,
        type_dataset: str,
    ):
        imgs_path = os.path.join(self.img_dir, type_dataset)
        imgs_labels = []
        for label in os.listdir(imgs_path):
            if label in self.classes:
                label_path = os.path.join(imgs_path, label)
                if os.path.isdir(label_path):
                    for img in os.listdir(label_path):
                        img_path = os.path.join(label_path, img)
                        label_tensor = tensor(self.classes.index(label))
                        imgs_labels.append((img_path, label_tensor))
        return imgs_labels

    def get_classes(self):
        return tuple(self.classes)
