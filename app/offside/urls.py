from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("index", views.index, name="index"),
    path("upload_image", views.upload_image, name="upload_image"),
    path("history", views.history, name="history"),
    path("train", views.train, name="train"),
]
