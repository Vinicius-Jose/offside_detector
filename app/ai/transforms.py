from PIL import Image
from torch import Tensor
from .ai_model import YOLOModel
from PIL import Image


class YOLOTransformer(object):

    def __init__(
        self,
        yolo_model: YOLOModel,
        conf: float = 0.4,
        batch: int = 1,
        save: bool = False,
        file_name: str = None,
    ) -> None:
        self.yolo_model = yolo_model
        self.conf = conf
        self.batch = batch
        self.save = save
        self.file_name = file_name

    def __call__(self, image: Tensor) -> dict:
        image = image.unsqueeze(0)
        result = self.yolo_model.predict(image, conf=self.conf, batch=self.batch)
        im_bgr = result[0].plot(labels=False, probs=False)
        im_rgb = Image.fromarray(im_bgr)
        if self.save:
            im_rgb.save(self.file_name)
        return im_rgb
