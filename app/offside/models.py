from django.db import models

# Create your models here.


class Prediction(models.Model):
    image = models.ImageField(null=False, upload_to="images/original_images/")
    predicted_image = models.ImageField(null=True, upload_to="images/predictions/")
    offside_prob = models.FloatField(null=True)
    date = models.DateTimeField(auto_now=True)
