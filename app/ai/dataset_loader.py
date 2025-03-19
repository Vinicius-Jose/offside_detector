from roboflow import Roboflow, Workspace, Project
from roboflow.core.dataset import Dataset
import os
import yaml


class DatasetLoader:

    def __init__(self) -> None:
        self.default_dataset_folder = os.getenv("DEFAULT_DATASET_FOLDER")
        self.api_key = os.getenv("ROBOFLOW_KEY")

    def roboflow_loader(
        self,
        workspace: str,
        project: str,
        location: str,
        version_number: int = 0,
        overwrite: bool = False,
        model_format: str = "yolov11",
    ) -> Dataset:
        rf = Roboflow(api_key=self.api_key)
        workspace: Workspace = rf.workspace(workspace)
        project: Project = workspace.project(project)
        if not version_number:
            version_number = project.get_version_information()
            version_number = version_number[0].get("id")
            version_number = version_number.split("/")[-1]

        version = project.version(version_number=version_number)
        dataset = version.download(
            model_format,
            location=self.default_dataset_folder + location,
            overwrite=overwrite,
        )
        self.__update_yaml_path(location)
        return dataset

    def __update_yaml_path(self, prefix: str) -> None:
        path = self.default_dataset_folder + prefix + "data.yaml"
        if os.path.isfile(path):
            with open(path, "r") as file:
                data = yaml.safe_load(file)
            data["path"] = f"./{prefix}"

            with open(path, "w") as file:
                yaml.safe_dump(data, file)
