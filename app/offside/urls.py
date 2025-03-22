from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("index", views.index, name="index"),
    path("upload_image", views.upload_image, name="upload_image"),
    path("history", views.history, name="history"),
    path("train", views.train, name="train"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
