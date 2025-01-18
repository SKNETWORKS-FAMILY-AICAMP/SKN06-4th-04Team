# chat/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import json
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)

PERSIST_DIRECTORY = os.path.abspath(os.path.join("chat", "vector_store"))
COLLECTION_NAME = "tax_law"

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

@csrf_exempt
def chatbot_view(request):
    """
    챗봇 HTML 페이지 렌더링
    """
    return render(request, 'chat/chatbot.html')

@csrf_exempt
def get_answer(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode('utf-8'))
            question = body.get("question", "").strip()

            if not question:
                return JsonResponse({"error": "질문이 비어 있습니다."}, status=400)

            if not retriever:
                return JsonResponse({"error": "벡터스토어가 초기화되지 않았습니다."}, status=500)

            related_docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in related_docs])
            
            messages = [
                {"role": "system", "content": "당신은 대한민국 세법에 대해 전문적인 지식을 가진 도우미입니다."},
                {"role": "user", "content": f"질문: {question}\n관련 정보:\n{context}"}
            ]

            response = chat_model(messages)

            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = "답변을 생성할 수 없습니다."

            return JsonResponse({"answer": answer})

        except Exception as e:
            print(f"오류 발생: {str(e)}") 
            return JsonResponse({"error": f"서버 오류 발생: {str(e)}"}, status=500)

    return JsonResponse({"error": "GET 요청은 지원되지 않습니다."}, status=405)
