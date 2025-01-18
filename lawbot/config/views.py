# config/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
import json
import requests
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from users.views import CustomLoginView
from users.forms import CustomAuthenticationForm
from django.contrib.auth import login, logout, update_session_auth_hash

# def chat_view(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             user_message = data.get('message')
#             history = data.get('history', [])

#             response = requests.post(
#                 'http://127.0.0.1:8000/chat/send_message/',  # AI 요청
#                 json={'message': user_message, 'history': history}
#             )
            
#             if response.status_code == 200:
#                 ai_response = response.json().get('response')
#                 history.append(('user', user_message))
#                 history.append(('AI', ai_response))
#                 request.session['history'] = history
#                 return JsonResponse({'response': ai_response})
#             else:
#                 return JsonResponse({'error': 'API 호출 실패'}, status=400)
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return render(request, 'chat/chatbot.html')

    
def home_view(request):
    if request.user.is_authenticated:
        # 인증된 사용자일 경우 챗봇 페이지로 이동
        return render(request, 'chat/chatbot.html')
    
    if request.method == 'POST':
        # 로그인 처리
        form = CustomAuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user) 
            return redirect('chat')  
    else:
        form = CustomAuthenticationForm()

    # 비인증 사용자에게 홈 페이지와 로그인 폼 제공
    return render(request, 'home.html', {'form': form})