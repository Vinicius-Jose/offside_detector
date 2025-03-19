from ai_model import YOLOModel
from dataset_loader import DatasetLoader
from dotenv import load_dotenv

load_dotenv("./.env")
#
#
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

model = YOLOModel()
model.train(epochs=1, dataset="player-detection/data.yaml")


result = model.predict(["./test_image.jpg", "test_2.jpg"], conf=0.01)
