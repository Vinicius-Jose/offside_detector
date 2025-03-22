from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from .models import Prediction

from ai.facade import AIFacade

from django.core.files.base import ContentFile
import base64
from threading import Thread
from datetime import datetime


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")


def upload_image(request: HttpRequest) -> HttpResponse:
    image = request.FILES
    prediction = Prediction(image=image.get("image"))
    prediction.save()
    ai = AIFacade()
    predicted = ai.predict(prediction.image.file)
    image = base64.b64decode(predicted["image_predicted_base64"])
    image_file = ContentFile(image, name=prediction.image.name.split("/")[-1])

    prediction.predicted_image = image_file
    prediction.offside_prob = predicted["offside"]
    result = predicted["result"]
    prediction.save()
    return render(
        request,
        "prediction.html",
        context={
            "prediction": prediction,
            "result": result,
            "prob": predicted[result] * 100,
        },
    )


def history(request: HttpRequest) -> HttpResponse:
    predictions = Prediction.objects.all()
    return render(request, "history.html", context={"predictions": predictions})


def train(request: HttpRequest) -> HttpResponse:
    ai = AIFacade()
    thread = Thread(target=ai.train)
    thread.start()
    return render(request, "index.html", context={"message": "Train started"})
