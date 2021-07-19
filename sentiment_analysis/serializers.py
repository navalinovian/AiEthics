from rest_framework import serializers
from .models import Sentiment, Implement

class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model= Sentiment
        fields =('id', 'text', 'label')

class ImplementSerializer(serializers.ModelSerializer):
    class Meta:
        model= Implement
        fields =('id', 'text', 'label')