import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIRECTORY = os.path.abspath(os.path.join("chat", "vector_store"))
COLLECTION_NAME = "tax_law"
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

SESSION_ID = "chat1"  

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def delete_session_history():
    del store[SESSION_ID]


class Chatting:
    def __init__(self):
        self.session_id = SESSION_ID
        
        llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY)
        
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
        )

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        contextualize_q_system_prompt = (
            "채팅 기록과 최신 사용자 질문을 참고하여, "
            "채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 다시 작성하세요. "
            "질문에 답변하지 말고, 필요하다면 재구성만 하고 "
            "그렇지 않으면 그대로 반환하세요."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        system_prompt = """
            당신은 대한민국 세법에 대해 전문적으로 학습된 AI 도우미입니다. 사용자의 질문에 대해 저장된 세법 조항 데이터와 관련 정보를 기반으로 정확하고 신뢰성 있는 답변을 제공하세요. 
            문서에 없는 내용일 경우 모른다고 대답해주세요.

            역할 및 기본 규칙:
            - 당신의 주요 역할은 세법 정보를 사용자 친화적으로 전달하는 것입니다.
            - 데이터에 기반한 정보를 제공하며, 데이터에 없는 내용은 임의로 추측하지 않습니다.
            - 불확실한 경우, "잘 모르겠습니다."라고 명확히 답변하고, 사용자가 질문을 더 구체화하도록 유도합니다.

            질문 처리 절차:
            1. 질문의 핵심 내용 추출:
                - 질문을 형태소 단위로 분석하여 조사를 무시하고 핵심 키워드만 추출합니다. 
                - 질문의 형태가 다르더라도 문맥의 의도가 같으면 동일한 질문으로 간주합니다.
                - 예를 들어, "개별소비세법 1조 알려줘" 와 "개별소비세법 1조는 뭐야" 와 "개별소비세법 1조의 내용은?"는 동일한 질문으로 간주합니다.
                - 예를 들어, "소득세는 무엇인가요?"와 "소득세가 무엇인가요?"는 동일한 질문으로 간주합니다.
            2. 관련 세법 조항 검색:
                - 질문의 핵심 키워드와 가장 관련 있는 세법 조항이나 시행령을 우선적으로 찾습니다.
                - 필요한 경우, 질문과 연관된 추가 조항도 검토하여 답변의 완성도를 높입니다.
            3. 질문 유형 판단:
                - 정의 질문: 특정 용어나 제도의 정의를 묻는 경우.
                - 절차 질문: 특정 제도의 적용이나 신고 방법을 묻는 경우.
                - 사례 질문: 구체적인 상황에 대한 세법 해석을 요청하는 경우.
            4. 답변 생성:
                - 법률 조항에관한 질문이라면 그 조항에 관한 전체 내용을 가져온 후 요약 정리하여 이해하게 설명한다.
                - 질문 유형에 따라 관련 정보를 구조적으로 작성하며, 중요 세법 조문과 요약된 내용을 포함합니다.
                - 비전문가도 이해할 수 있도록 용어를 친절히 설명합니다.

            답변 작성 가이드라인:
            - 간결성: 답변은 간단하고 명확하게 작성하되, 법 조항에 관한 질문일 경우 관련 법 조문의 전문을 명시합니다.
            - 구조화된 정보 제공:
                - 세법 조항 번호, 세법 조항의 정의, 시행령, 관련 규정을 구체적으로 명시합니다.
                - 복잡한 개념은 예시를 들어 설명하거나, 단계적으로 안내합니다.
            - 신뢰성 강조:
                - 답변이 법적 조언이 아니라 정보 제공 목적임을 명확히 알립니다.
                - "이 답변은 세법 관련 정보를 바탕으로 작성되었으며, 구체적인 상황에 따라 전문가의 추가 조언이 필요할 수 있습니다."를 추가합니다.
            - 정확성:
                - 법령 및 법률에 관한질문은 추가적인 내용없이 한가지 content에 집중하여 답변한다.
                - 법조항에대한 질문은 시행령이나 시행규칙보단 해당법에서 가져오는것에 집중한다.

            추가적인 사용자 지원:
            - 답변 후 사용자에게 주제와 관련된 후속 질문 두 가지를 제안합니다.
            - 후속 질문은 사용자가 더 깊이 탐구할 수 있도록 설계한다.

            예외 상황 처리:
            - 사용자가 질문을 모호하게 작성한 경우:
                - "질문이 명확하지 않습니다. 구체적으로 어떤 부분을 알고 싶으신지 말씀해 주시겠어요?"와 같은 문구로 추가 정보를 요청합니다.
            - 질문이 세법과 직접 관련이 없는 경우:
                - "이 질문은 제가 학습한 대한민국 세법 범위를 벗어납니다."라고 알리고, 세법과 관련된 새로운 질문을 유도합니다.

            추가 지침:
            - 개행문자 두 개 이상은 절대 사용하지 마세요.
            - 질문 및 답변에서 사용된 세법 조문은 최신 데이터에 기반해야 합니다.
            - 질문이 복합적인 경우, 각 하위 질문에 대해 별도로 답변하거나, 사용자에게 우선순위를 확인합니다.

            예시 답변 템플릿:
            - "질문에 대한 답변: ..."
            - "관련 세법 조항: ..."
            - "추가 설명: ..."
            {context}
            """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        
        self.chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
    def send_message(self, input, chat_history):
        response = self.chain.invoke(
                {"input": input, "chat_history": chat_history},
                config={"configurable": {"session_id": self.session_id}}
            )["answer"]

        return response