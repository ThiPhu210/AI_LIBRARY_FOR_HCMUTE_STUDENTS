from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('', views.chat_with_bot, name='chat-with-bot'),
]