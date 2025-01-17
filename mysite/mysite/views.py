# mysite/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
import requests
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        # POST 요청 처리
        try:
            data = json.loads(request.body)
            user_message = data.get('message')
            history = data.get('history', [])

            response = requests.post(
                'http://127.0.0.1:8000/chat/send_message/',  # AI 요청
                json={'message': user_message, 'history': history}
            )
            
            if response.status_code == 200:
                ai_response = response.json().get('response')
                history.append(('user', user_message))
                history.append(('AI', ai_response))
                request.session['history'] = history
                return JsonResponse({'response': ai_response})
            else:
                return JsonResponse({'error': 'API 호출 실패'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    # GET 요청 처리 (채팅 페이지 렌더링)
    return render(request, 'users/chat.html')



def home_view(request):
    return render(request, 'users/home.html')  # home.html 템플릿을 렌더링