# OFFSIDE DETECTOR
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](./LICENSE)

## What is this project?
 - The purpouse of this project is create an AI using Pytorch and YOLO to decide if an image of a football match is offside or onside.

## What was the technologies used?
 -  The project was created using:
    - Pytorch
    - YOLO
    - Roboflow (datasets management only)
    - Django (Web interface and Database)
    - Poetry (dependencies management)

## How it works?
 - Every time an image is upload to the system, the image goes first to an YOLO model to identify the ball, players and refeere and with the result image pass trough a Pytorch model to decide if is offside or onside
 - The project was trained with to datasets, one for YOLO and another for Pytorch, both datasets its available on Roboflow, and if you click "Train" the system will download the datasets before start training (WARNING!: you will need a Roboflow account and a Roboflow account, see this [link](https://docs.roboflow.com/api-reference/authentication)) :
    - You can find the dataset for YOLO model [here](https://universe.roboflow.com/nikhil-chapre-xgndf/detect-players-dgxz0) (Note: this dataset was created for another user and maybe not be available forever)
    - You can find the dataset for Pytorch [here](https://universe.roboflow.com/football-sioqa/offside-dnedg
    )
    - The project already has a model trained inside the  app/ai/resources foulder, if you want to train from zero, just delete all files inside this directory and click Train on web page. Note: if you don't delete those files, the system will load them and continue the train, this may cause overfitting

## How to run the system?
- You can run like any other Django Application, but firstly install poetry, on your cli:
    ```bash
    pip instal poetry
    ```
- Then inside the folder where you downloaded this project run:
     ```bash  
     poetry install
    ```
    this command will create an virtual environment and install all dependencies required to run
- To run, you will need first start the database as any Django project, so on your cli :
    ```bash
    poetry run python app/manage.py makemigrations offside
    poetry run python app/manage.py makemigrations 
    poetry run python app/manage.py migrate
    ```
- Now you only need to start the system:
    ```bash
    poetry run python app/manage.py runserver
    ```
- For more information on how to use Poetry please visit this [link](https://python-poetry.org/docs/basic-usage/)


## System in execution:
- You only have two pages and an additional option:
    - Index: where you upload image and send them to be predicted.
    - History: where you can see all images predicted 
    - Train: this will start training the model in a thread , please note that this may cause performance issues on your computer
