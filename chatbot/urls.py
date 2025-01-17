from django.urls import path
from chatbot.views import chatbot_page, get_answer  # views.py에서 함수 가져오기

urlpatterns = [
    path('', chatbot_page, name='chatbot_page'),       # 메인 페이지
    path('get-answer', get_answer, name='get_answer')  # 챗봇 응답 처리 URL
]
