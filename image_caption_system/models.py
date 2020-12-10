from django.db import models


# Create your models here.
class StyleImageCaption(models.Model):

    image_id = models.CharField(max_length=200)
    image_path = models.CharField(max_length=200)
    time_stamp = models.CharField(max_length=200)
    caption = models.CharField(max_length=200)
    info = models.CharField(max_length=1000)

    def __str__(self):
        return self.image_path
