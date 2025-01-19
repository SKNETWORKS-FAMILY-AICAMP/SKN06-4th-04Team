# chat/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .llm import Chatting, delete_session_history, store, SESSION_ID, get_session_history
import json


chain=Chatting()

@csrf_exempt
def chatbot_view(request):
    # history = get_session_history(SESSION_ID)
    
    return render(request, 'chat/chatbot.html',)

@csrf_exempt
def get_answer(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body) 
            question = body.get("question", "").strip()
            history = get_session_history(SESSION_ID)

            response =  chain.send_message(input=question, chat_history=history)
            
            return JsonResponse({"answer": response}, status=200)
            
        except Exception as e:
            print(f"오류 발생: {str(e)}") 
            return JsonResponse({"error": f"서버 오류 발생: {str(e)}"}, status=500)

    return JsonResponse({"error": "GET 요청은 지원되지 않습니다."}, status=405)

@csrf_exempt
def new_chat(request):
    if request.method == "POST":
        try:
            delete_session_history()
            
            return JsonResponse({"resultMsg": '새로운 대화를 시작합니다.'}, status=200)
            
        except Exception as e:
            print(f"오류 발생: {str(e)}") 
            return JsonResponse({"error": f"서버 오류 발생: {str(e)}"}, status=500)

    return JsonResponse({"error": "GET 요청은 지원되지 않습니다."}, status=405)