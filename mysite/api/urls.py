from django.urls import path
from .views import  chat_message

urlpatterns = [
    path('chat_message/<str:message>/', chat_message, name='chat_message'),
]
# api/chat_message/안녕하세요.