from django.contrib import admin
from django.urls import path
from .views import SentimentView, sentiment_data_upload_view 
from .views import sentiment_data_view, sentiment_data_preprocessing, sentiment_evaluate, youtube_api

urlpatterns = [
    # path('home', RoomView.as_view()),
    path('sentiment', SentimentView.as_view()),
    path('', sentiment_data_upload_view, name='upload'),
    path('data-view', sentiment_data_view, name='data_view'),
    path('data-preprocessing', sentiment_data_preprocessing, name='data_preprocessing'),
    path('evaluate', sentiment_evaluate, name='evaluate'),
    path('evaluate/<int:surrogate>/', sentiment_evaluate, name='evaluate'),
    path('youtube-api', youtube_api, name='youtube'),
]