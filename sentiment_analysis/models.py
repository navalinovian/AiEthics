from django.db import models

class Sentiment(models.Model):
    text        = models.TextField()
    label       = models.CharField(max_length=8, default="", unique=False)