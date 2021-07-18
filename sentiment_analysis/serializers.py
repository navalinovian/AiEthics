from rest_framework import serializers
from .models import Sentiment

class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model= Sentiment
        fields =('id', 'text', 'label')