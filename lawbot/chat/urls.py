from django.urls import path
from chat.views import chatbot_view, get_answer

urlpatterns = [
    path('', chatbot_view, name='chat'),
    path('chat/get-answer', get_answer, name='get_answer')
]
