# 💸 Chatbot - 대한민국 세법 질의응답 챗봇 💸

## 📍목차
1. [팀원 소개](#1️⃣-팀원-소개)
2. [프로젝트 소개](#2️⃣-프로젝트-소개)
   - [디렉토리 구조](#디렉토리-구조)
   - [주제](#주제)
   - [주제를 선택한 이유](#주제를-선택한-이유)
   - [주요 기능](#주요-기능)
3. [AI모델 개선사항](#3️⃣-AI모델-개선사항)
3. [요구사항 정의서](#4️⃣-요구사항-정의서)
4. [화면 설계서](#5️⃣-화면-설계서)
5. [시스템 구성도](#6️⃣-시스템-구성도)
6. [테스트 계획서 및 테스트 결과 보고서](#7️⃣-테스트-계획서-및-테스트-결과-보고서)
7. [📝한 줄 회고](#📝-한-줄-회고)

---

<br/>

## 1️⃣ 팀원 소개


| 이름  | 담당 업무  |  |
|-----|------|----------|  
|박유나|프로젝트 관리, Django BE, FE, 화면설계, 요구사항 정의서, RAG 모델, 문서작성|
|유경상|Django BE (chat 관련), 데이터 처리 및 RAG 모델 개선, 테스트|
|노원재|Django BE (chat 관련), 테스트|
|전하연|Django BE (user 관련), 요구사항 정의서|
|박서윤|Django BE (user 관련), 요구사항 정의서|
<br/>

## 2️⃣ 프로젝트 소개

### 디렉토리 구조
```
├── data/                                    # PDF 데이터 (75개)
├── lawbot/                                  # 📍Django
│    ├── chat/                               # app: 챗봇
│    │    ├── template/chatbot.html
│    │    ├── vector_store                   # vector store
│    │    ├── ...
│    │    ├── llm.py                         # chain
│    │    ├── urls.py
│    │    └── views.py
│    ├── config/              
│    │    ├── settings.py
│    │    ├── urls.py 
│    │    └── views.py
│    ├── media/profile_pics/                 # 프로필사진
│    ├── static/                             # 정적 파일
│    │    ├── css
│    │    ├── img
│    │    └── js
│    ├── templates/                          # html
│    │        ├── home.html
│    │        └── layout.html
│    └── users/                              # app: 회원
│         ├── template/
│         │   ├── profile.html
│         │   └── register.html
│         ├── forms.py                       # Form 관련
│         ├── models.py        
│         ├── urls.py
│         └── views.py
├── report/                                  # 산출물 및 readme.md image
├── .gitignore            
├── loader.py                                # 모델 생성 함수 모듈   
├── main.ipynb                               # 모델 생성         
└── readme.md  
```

### 주제
- 대한민국 세법 질의응답 챗봇을 Django 기반 웹 애플리케이션으로 구현.

### 주제를 선택한 이유
1. 세법은 일반적으로 잘 알기 어려운 분야라서, 사용자들에게 실질적인 도움을 줄 수 있을 것이라 생각했습니다.
2. 팀원들 역시 세법 정보가 필요했던 경험이 있어 이 주제에 더 흥미를 느꼈습니다.
3. 데이터 수집과 전처리를 통해 많은 것을 배울 수 있을 것이라 판단했고, 실제 회사 문서와 유사한 형태의 데이터라서 학습과 활용 면에서 적합하다고 여겼습니다.
4. 배운 내용을 다양한 방식으로 적용할 수 있어 복습에 좋은 데이터라고 생각했으며, 데이터의 양도 충분히 많아 기본에 충실하면서도 확장성 있게 학습할 수 있을 것 같았습니다.

### 주요 기능
1. **회원 관리**
   - 회원가입 및 로그인 기능.
   - 사용자 정보 수정 및 관리.

2. **챗봇 질의응답**
   - 연말정산, 세법에 대한 질의응답.
   - Session 기반 대화 히스토리 저장하여 맥락 기반 응답 제공

3. **RAG 기반 모델 개선**
   - Chroma Vectorstore 및 chunk 기반 데이터 스플릿을 통한 응답 정확도 향상.
   - RAGas 도입


<br/>

## 3️⃣ RAG 모델 개선사항

 <i>* 이전 프로젝트 → [Github Repository](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-4Team) </i>
### 이전 모델의 문제점
- embedding vector의 차원수 초과 -> chunk사이즈 및 overlap 수치저하로 인한 문맥파악능력 저하
- 세법 데이터를 각 조항별로 split 하였음 → 컨텍스트 부족으로 인한 부정확한 응답 발생.
   <br/>
   - 예) "개별소비세법 <u>제1조</u>를 알려줘!" 라는 질문에 응답하기 어려워 함. 할루시네이션 발견견.
   <br/> 
   → 각 세법에 따라 법률, 시행령, 시행규칙을 잘 구분하지 못하기 때문이라고 판단.
      - 개별소비세법 <u>법률 제1조</u> 
      - 개별소비세법 <u>시행령 제1조</u>
      - 개별소비세법 <u>시행규칙 제1조</u>
   
   - 해결책: chunk기반으로 split하여 문맥을 읽히기로 함.
      - cf) prompt 로 해결해 보고자 하였으나 정확도가 낮음
      - 72개의 파일을 7개의 폴더로 나눈 후 각 폴더별 embedding 후 빈 vector store에 차례대로 추가 및 저장.
        
   - 결과: 매우 성공적!

- 성능 평가 미흡
   - precision, recall, relevency 확인시 정답내용을 임의적으로 선별함.
   - 평가 데이터 부족, 몇가지 조항들로만 LLM모델 평가
 
   - 해결책:
      - RAGas평가 도입(평가용 chain을 구성 -> 평가 데이터로 사용할 context를 추출 -> 평가데이터셋 구성 -> RAGas평가)
      - Hometax Q&A 와 LLM모델의 응답 비교
      - LLM모델의 응답과 Hometax Q&A정답간의 문맥 유사도 비교
      - LLM모델의 응답과 Hometax Q&A정답간의 BLEU, ROUGE Score 를 통한 평가

### 개선 내용
1. **Chunk 기반 데이터 스플릿**
   - 데이터를 문맥 단위로 분리하여 질의응답 정확성 향상.
   - 응답의 일관성과 신뢰도 15% 이상 증가.

2. **RAGas 도입**
   - 응답 속도 최적화 및 사용자의 요구에 따른 적응형 모델 구조.
   - Q-A질문 쌍이 주어진 context에 얼마나 부합한지 재현율, 정밀도, 신뢰성, 적합성

3. **벡터스토어 최적화**
   - Chroma Vectorstore의 효율적인 인덱싱 및 쿼리 처리 개선.

4. **다양한 평가지표 도입**
   - RAGas를 통한 context
   - 홈텍스 내의 질의응답 게시판과 LLM모델의 답변 비교
   - cos 유사도를 통한 질문과 LLM답변간의 문백 유사도 검증
   - bleu스코어 및 rouge 스코어 검증

```
### RAG 구현
- 질의 처리 로직 구현
- 관련 정보 검색 메커니즘 구축
  ```python3
  # Prompt Template 생성
  messages = [
      ("ai", """
      당신은 대한민국 세법에 대해 전문적으로 학습된 AI 도우미입니다. 사용자의 질문에 대해 저장된 세법 조항 데이터와 관련 정보를 기반으로 정확하고 신뢰성 있는 답변을 제공하세요. 

      **역할 및 기본 규칙**:
      - 당신의 주요 역할은 세법 정보를 사용자 친화적으로 전달하는 것입니다.

    ...
```
```
# 평가용 chain  구성
class EvalDatasetSchema(BaseModel):
    user_input:str = Field(..., description="질문(Question)")
    retrieved_contexts:list[str] = Field(..., description="LLM이 답변할 때 참조할 context")
    reference: str = Field(..., description="정답(ground truth)")

jsonparser = JsonOutputParser(pydantic_object=EvalDatasetSchema)

qa_prompt_template = PromptTemplate.from_template(
    template=dedent("""
        당신은 RAG 평가를 위해 질문과 정답 쌍을 생성하는 인공지능 비서입니다.
        다음 [Context] 에 문서가 주어지면 해당 문서를 기반으로 {num_questions}개의 질문을 생성하세요. 

        질문과 정답을 생성한 후 아래의 출력 형식 GUIDE 에 맞게 생성합니다.
        질문은 반드시 [context] 문서에 있는 정보를 바탕으로 생성해야 합니다. [context]에 없는 내용을 가지고 질문-답변을 절대 만들면 안됩니다.
        질문은 간결하게 작성합니다.
        하나의 질문에는 한 가지씩만 내용만 작성합니다. 
        질문을 만들 때 "제공된 문맥에서", "문서에 설명된 대로", "주어진 문서에 따라" 또는 이와 유사한 말을 하지 마세요.
        정답은 반드시 [context]에 있는 정보를 바탕으로 작성합니다. 없는 내용을 추가하지 않습니다.
        질문과 답변을 만들고 그 내용이 [context] 에 있는 항목인지 다시 한번 확인합니다.
        생성된 질문-답변 쌍은 반드시 dictionary 형태로 정의하고 list로 묶어서 반환해야 합니다.
        질문-답변 쌍은 반드시 {num_questions}개를 만들어 주십시오.
                    
        출력 형식: {format_instructions}

        [Context]
        {context}
        """
    ),
    partial_variables={"format_instructions":jsonparser.get_format_instructions()}
)

# 데이터셋 생성 체인 구성
model = ChatOpenAI(model="gpt-4o")
dataset_generator_chain = qa_prompt_template | model | jsonparser
```
```
# 임의로 10개의 q-a쌍 추출
## 5개의 context 임의 추출
## 각 context 당 2개의 q-a쌍 생성.
total_samples = 5

idx_list = list(range(len(all_documents)))
random.shuffle(idx_list)

eval_context_list = []
while len(eval_context_list) < total_samples:
    idx = idx_list.pop()
    context = all_documents[idx].page_content
    if len(context) > 100:
        eval_context_list.append(context)

eval_data_list = []
num_questions = 2
for context in eval_context_list:
    _eval_data_list = dataset_generator_chain.invoke(
        {"context":context, "num_questions":num_questions}
    )
    for eval_data in _eval_data_list:
        eval_data['retrieved_contexts'] = [context]
    
    eval_data_list.extend(_eval_data_list)

eval_df = pd.DataFrame(eval_data_list)

context_list = []
response_list = []

for user_input in eval_df['user_input']:
    res = rag_chain.invoke(user_input)
    context_list.append(res['source_context'])
    response_list.append(res['llm_answer'])

eval_df["retrieved_contexts"] = context_list
eval_df["response"] = response_list

eval_dataset = EvaluationDataset.from_pandas(eval_df)
```
```
# 평가모델 wrapping
load_dotenv()
model = ChatOpenAI(model= "gpt-4o")
eval_llm = LangchainLLMWrapper(model)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
eval_embedding = LangchainEmbeddingsWrapper(embedding_model)
metrics = [
    LLMContextRecall(llm=eval_llm),
    LLMContextPrecisionWithReference(llm=eval_llm),
    Faithfulness(llm=eval_llm),
    AnswerRelevancy(llm=eval_llm, embeddings=eval_embedding)
]

# BLEU, ROUGE Score 검증
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
bleu_score = sentence_bleu([ground_truth.split()], answer.split())
rouge_scores = scorer.score(answer, ground_truth)

# 결과
result = evaluate(dataset=eval_dataset, metrics=metrics)
result.to_pandas()
print(f"BLEU점수:{bleu_score:.2f}")
print(f"Rouge1점수:{rouge_scores['rouge1']}")
print(f"RougeL점수:{rouge_scores['rougeL']}")
```
- 결과<br/>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team/blob/main/result_img/q_a_10pair.png"> <br/>

- 결과<br/>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team/blob/main/result_img/hometax.png"><br/>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team/blob/main/result_img/hometax_response.png"><br/>
<br/>
답변: 
 취업 전 지출한 월세나 신용카드 결제 금액의 연말정산 공제 여부는 특정 조건에 따라 달라질 수 있습니다.일반적으로 연말정산에서 소득공제는 해당 과세기간 내 소득이 있는 경우에만 공제 가능합니다.따라서, 취업 이전에 소득이 없다면 해당 기간 동안의 지출에 대해서는 소득공제를 받을 수 있는 기준이 충족되지 않을 수 있습니다.
1. 월세 공제: 월세 세액공제는 근로소득이 있는 경우 해당 과세기간에 실제로 납부한 월세에 대해 적용됩니다. 따라서, 취업 전 월세 지출은 공제 대상이 아닐 가능성이 높습니다.
2. 신용카드 공제: 신용카드 사용금액 소득공제는 근로소득이 있는 거주자가 해당 과세연도에 사용한 금액에 대해 적용됩니다.취업 전 사용한 금액은 소득이 없었기 때문에 공제 대상이 아닐 수 있습니다.
각 개인의 상황에 따라 다를 수 있으므로, 정확한 판단을 위해서는 세무 전문가와 상담하거나 국세청의 지침을 확인하는 것이 좋습니다. 이 답변은 세법 관련 정보를 바탕으로 작성되었으며, 구체적인 상황에 따라 전문가의 추가 조언이 필요할 수 있습니다. <br/>

- 결과<br/>
Semantic Similarity: 0.59 <br/><br/>
BLEU점수:0.00 <br/><br/>
Rouge1점수:Score(precision=0.2, recall=0.5, fmeasure=0.28571428571428575) <br/>
RougeL점수:Score(precision=0.2, recall=0.5, fmeasure=0.28571428571428575) <br/>
</br>
- **결과해석**
     : answer rerlevancy 부분이 안좋게 평가가나왔다. 하지만 홈텍스Q&A 와 LLM모델의 Response가 같은 내용이었으나 문맥유사도 및 BLEU 점수가 0인것으로 보아 아직 같은 내용이어도 평가 모델들이 두 정답간의 내용적 의미를 잘 파악하지못하는것으로 보아 아직 모듈에서 제공하는 평가모델들이 문맥유사도 및 내용의 유사성을 잘 파악 못하는것 같다.

## 4️⃣ 요구사항 정의서

📍 <i>자세한 <u>**&lt;요구사항 정의서&gt;**</u>  문서 확인 → [Google sheet](https://docs.google.com/spreadsheets/d/1wKG5qj-1ep-ace8Mv8cm0a9_KxEt96GTHUgf0Eu_epk/edit?gid=0#gid=0) 참고 </i>

### 필수 요구사항
- 사용자 인증 및 권한 관리.
- 회원정보 수정
- 가입한 회원은 프로필 사진 등록/수정/삭제
- 인증 된 사용자에 한하여 챗봇 기능 이용
- 세법 관련 벡터 데이터베이스와 LLM을 연동하여 응답 생성
- 대화 히스토리 저장하여 맥락 기반 응답 제공


### 선택적 요구사항
- 반응형 UI 및 사용자 친화적 인터페이스.


<br/>

## 5️⃣ 화면 설계서

📍 <i>자세한 <u>**&lt;화면 설계서&gt;**</u>  문서 확인 → [Figma](https://www.figma.com/design/lHj5Rco1lt6BvE4YdrM6ys/SKN06_4th_T4_%ED%99%94%EB%A9%B4%EC%84%A4%EA%B3%84%EC%84%9C?node-id=0-1&t=iVbEYIeSSG7Xd9VC-1) 참고 </i>

![alt text](</result_img/화면설계서.png>)
1. **로그인 페이지**
   - url : ' / ' 

2. **챗봇 대화 페이지 (로그인시)**
   - url : ' / '

3. **회원가입 페이지**
   - url : ' /register '
   - 아이디, 비밀번호, 비밀번호 확인, 이름, 생년월일, 이메일, 휴대폰번호

4. **내 프로필**
   - url : ' /profile '

5. **내 프로필 수정**
   - url : ' /profile/update '
   - 프로필 사진, 이름, 생년월일, 이메일, 휴대폰번호, 비밀번호, 비밀번호 확인

<br/>

## 6️⃣ 시스템 구성도

```
⚠️ 준비중 ...
```

<br/>

## 테스트 계획서 및 테스트 결과 보고서

📍 <i><u>**&lt;테스트 계획서&gt; 및 &lt;테스트 결과 보고서&gt;**</u>  → [Google sheet](https://docs.google.com/spreadsheets/d/1wKG5qj-1ep-ace8Mv8cm0a9_KxEt96GTHUgf0Eu_epk/edit?gid=1302105477#gid=1302105477) 참고 </i>

### 테스트 계획서

1. **회원 관리 테스트**  

   - 회원가입: validation 테스트.
   - 로그인: validation 테스트.
   - 정보 수정: 권한 확인 및 validation 테스트.
   - ...
2. **챗봇 기능 테스트**
   
   - 다양한 질의응답 시나리오에 대한 정확성 테스트.
   - Session 기반 히스토리 동작 확인.
   - '새로운 대화 시작하기' 기능 관련 테스트
      - history 초기화
      - 말풍선 지우기
      - 챗봇 Intro 생성
      - UI 테스트(스크롤 등..)
   - 반응형 UI.
   - 챗봇 키워드 카테고리 클릭시 응답 생성
   - 챗봇 요청중 일때 input 비활성화하여 이중 호출 방지
   - ...

### 테스트 결과 보고서
- **회원 관리**: 모든 시나리오 정상 동작 확인.
- **챗봇 응답**: chunk 기반 스플릿 도입 이후 응답 정확도 15% 향상.



## 📝 한 줄 회고
 - **노원재**: 

 - **박서윤**: 

 - **박유나**: 

 - **유경상**: 
 
 - **전하연**: 
