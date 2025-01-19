
SKN06-4TH-04Team


# 👩‍⚖️LLM 을 연동한 내 외부 문서 기반 질의 응답 웹페이지 개발👨‍⚖️ 
 
*개발기간:* 2025.01.16 - 2025.01.20
=======


# 👩‍⚖️세법 질의응답 시스템👨‍⚖️ 
 


## 💻 팀 소개

# 팀명: 절세미인

<img src="https://github.com/user-attachments/assets/df44c785-4e29-44ce-ab52-48ed3abee8d7" width="600" height="500">



| [노원재] | [박서윤] | [박유나] | [유경상] | [전하연] |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|<img src="https://github.com/user-attachments/assets/83778ae7-cead-4f34-ab0c-27d50300b23f" width="160" height="160">|<img src="https://github.com/user-attachments/assets/0889164e-66b0-4df2-b3f7-4ede25b6ee80"  width="160" height="160">|<img src="https://github.com/user-attachments/assets/faf2d9c0-a945-4604-98bb-af2b1907acae"   width="160" height="160">|<img src="https://github.com/user-attachments/assets/bd07a4f7-65aa-49c2-9cfd-e54f41d08282"  width="160" height="160">|<img src="https://github.com/user-attachments/assets/087fcd42-4884-425d-a7b0-fe866ff18b03"  width="160" height="160">|
|    절세남    |     절세미녀    |    절세미녀     |     절세남    |     절세미녀    |


---

<<<<<<< HEAD
# 💻세법 질의응답 웹서비스💻

## 📌 1. 프로젝트 개요

### 1.1. 개발 동기 및 목적  
본 프로젝트는 **복잡한 세법 관련 질문에 대한 신속하고 정확한 답변 제공**을 목표로 하는 **세법 질의응답 웹서비스**을 구현하고자 했습니다. 법제처에서 제공하는 세법 관련 데이터를 활용하여 세금 관련 질의에 대한 정확한 답변을 제공하는 챗봇 시스템을 개발했으며, [SKN06-3rd-04team](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-4Team)을 기반으로 개발된 프로젝트입니다.

### 1.2. 필요성  
세법은 복잡하고 자주 개정되어 일반인들이 이해하기 어려운 경우가 많습니다. 특히 **개인과 기업의 세금 관련 문의**에 대해 신속하고 정확한 정보 제공이 필요합니다. 따라서, **세금 관련 자주 묻는 질문**에 즉시 답변할 수 있는 챗봇 웹페이지을 통해 **세법에 대한 접근성을 향상**시키고 사용자들의 이해를 돕고자 했습니다.

### 1.3. 개발 목표  
- **정확한 정보 제공:** 최신 세법 규정과 절차 반영  
- **정확성 향상:** 일반적인 챗봇 한계를 넘는 **구체적이고 정확한 세법 관련 답변 제공**
- **사용자 경험 개선:** 직관적이고 편리한 UI/UX 설계

## 📌 2. 요구사항 정의서

## 📌 3. 화면 설계서

## 📌 4. 시스템 구성도

## 📌 5. 테스트 계획서 및 테스트 결과 보고서

## 📌 6. 한 줄 회고
 - **노원재**: 
 - **박서윤**: 
 - **박유나**: 
 - **유경상**: 
 - **전하연**: 
=======
## 📌 1. 프로젝트 개요

### 1.1. 개발 동기 및 목적  
본 프로젝트는 **복잡한 세법 관련 질문에 대한 신속하고 정확한 답변 제공**을 목표로 하는 **세법 확인 챗봇 시스템**을 구현하고자 했습니다. 법제처에서 제공하는 세법 관련 데이터를 활용하여 세금 관련 질의에 대한 정확한 답변을 제공하는 챗봇 시스템을 개발했습니다.  

### 1.2. 필요성  
세법은 복잡하고 자주 개정되어 일반인들이 이해하기 어려운 경우가 많습니다. 특히 **개인과 기업의 세금 관련 문의**에 대해 신속하고 정확한 정보 제공이 필요합니다. 따라서, **세금 관련 자주 묻는 질문**에 즉시 답변할 수 있는 챗봇 시스템을 통해 **세법에 대한 접근성을 향상**시키고 사용자들의 이해를 돕고자 했습니다.

### 1.3. 개발 목표  
- **정확한 정보 제공:** 최신 세법 규정과 절차 반영  
- **정확성 향상:** 일반적인 챗봇 한계를 넘는 **구체적이고 정확한 세법 관련 답변 제공**  

### 1.4. 디렉토리 구조
```
├── data 
│   ├── tax_law1 : 세법 관련 (12개)
│       ├── 개별소비세법.pdf  
│       ├── 개별소비세법_시행규칙.pdf
│       ├── 개별소비세법_시행령.pdf  
│       └── ...
│   └── tax_law2 : 세법 관련(12개)
│       ├── 국세기본법.pdf
│   └── ...       
│   └── tax_law6 : 세법 관련
│       └── ...
│
│   └── tax_etc : 기타 데이터 (5개)
│       ├── 2024_핵심_개정세법.pdf  
│       ├── 연말정산_Q&A.pdf  
│       ├── 연말정산_신고안내.pdf  
│       ├── 연말정산_주택자금·월세액_공제의이해.pdf  
│       └── 주요_공제_항목별_계산사례.pdf  
├── loader.py : 데이터 로딩을 위한 함수 모듈  
└── main.ipynb : 프로젝트 최종 노트북 파일  
```

---

## 📌 2. 상세 내용

### 2.1. 데이터 수집
-  [법제처](https://www.law.go.kr/LSW/main.html)와 [국세청](https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2304&cntntsId=238938)에서 세법 관련 데이터 다운로드

### 2.2. 데이터 로드
- **PyMuPDFLoader**
  - 필요없는 페이지 제외하고 읽기

    ```python3
    text_loader = PyMuPDFLoader(pdf_file)
    texts = text_loader.load()

    for i, text in enumerate(texts):
        text.metadata["page"] = i + 1      

    page_ranges = [(17, 426)]
    texts = [
        text for text in texts
        if any(start <= text.metadata.get("page", 0) <= end for start, end in page_ranges)
    ]
    ```
- **Tabula**
  - PDF의 테이블 데이터 읽기용
  - 적용 파일:<br/>2024_핵심_개정세법.pdf, 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf, 주요_공제_항목별_계산사례.pdf

    ```python3
    tables = read_pdf(pdf_file, pages=table_pages_range, stream=True, multiple_tables=True)
        table_texts = [table.to_string(index=False, header=True) for table in tables]
    ```
### 2.3. 데이터 전처리
- **불필요 텍스트 제거**
  - 세법 관련 (개별소비세법~증권거래세법_시행령).pdf, 연말정산_Q&A.pdf
    ##### - 머리말, 꼬리말 제거
    ```python3
    rf'법제처\s*\d+\s*국가법령정보센터\n{file_name.replace('_', ' ')}\n'
    ```
    ##### - [ ]로 감싸진 텍스트 제거
    ```python3
    r'\[[\s\S]*?\]'
    #text = re.sub(r'【.*?】', '', text, flags=re.MULTILINE)
    ```
    ##### - < >로 감싸진  텍스트 제거
    ```python3
    r'<[\s\S]*?>'  
    ```
    ##### - 페이지 번호 패턴 제거
    ```python3
    text = re.sub(r'^- \d{1,3} -', '', text, flags=re.MULTILINE)
    ```
    
  - 2024_핵심_개정세법.pdf, 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf
    ##### - 머리말 및 사이드바 제거
    ```python3
    - r"2\n0\n2\n5\n\s*달\n라\n지\n는\n\s*세\n금\n제\n도|"  
    - r"\n2\n0\n2\n4\n\s*세\n목\n별\n\s*핵\n심\n\s*개\n정\n세\n법|"
    - r"\n2\n0\n2\n4\n\s*개\n정\n세\n법\n\s*종\n전\n-\n개\n정\n사\n항\n\s*비\n교\n|"
    - r"\s*3\s*❚국민･기업\s*납세자용\s*|"
    - r"\s*2\s*0\s*2\s*4\s|"
    - r"\s한국세무사회\s|" 
    - r"\n7\n❚국민･기업 납세자용|"
    - r"\n71\n❚상세본|"
    ```
    ##### - 문장이 다음줄로 넘어가면서 생기는 \n 제거 
    ```python3
    r"([\uAC00-\uD7A3])\n+([\uAC00-\uD7A3])"
    re.sub(pattern2, r"\1\2" , edit_content) #앞뒤글자 합치기
    ```

  - 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf, 주요_공제_항목별_계산사례.pdf
    ##### - NaN 제거
    ```python3
    r"\bNaN\b"
    ```
    #### - 하나 이상의 공백문자를 한 개의 공백문자로 바꾸기
    ```python3
    - r"\s+"
    - re.sub(pattern3, " ", edit_content)
    ```
    
    
### 2.4. split 방법
- 세법 관련 (개별소비세법~증권거래세법_시행령).pdf
  ```python3
  chunk size = 1000, over lap = 100 으로 설정
  ```

- 2024_핵심_개정세법.pdf, 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf, 주요_공제_항목별_계산사례.pdf
  ```python3
  chunk size = 1000, over lap = 100 으로 설정
  ```

- 연말정산_Q&A.pdf
  ```python3
   chunk size = 1000, over lap = 100 으로 설정
  ```



### 2.5. 벡터 데이터베이스 구현 
- ebedding vector의 차원수 초과로 72개의 파일을 6개의 폴더로 배분
- 법령 및 시행령 전처리, 표가 많은 etc폴더 파일들 전처리
- 각 폴더별 벡터화 후 빈 vector store에 저장.
- 선택한 벡터 데이터베이스에 데이터 저장
  ```python3
  COLLECTION_NAME = "tax_law"
  PERSIST_DIRECTORY = "tax"

  def set_vector_store(documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )
    ```

### 2.6. RAG 구현
- 질의 처리 로직 구현
- 관련 정보 검색 메커니즘 구축
  ```python3
  # Prompt Template 생성
  messages = [
      ("ai", """
      당신은 대한민국 세법에 대해 전문적으로 학습된 AI 도우미입니다. 사용자의 질문에 대해 저장된 세법 조항 데이터와 관련 정보를 기반으로 정확하고 신뢰성 있는 답변을 제공하세요. 

      **역할 및 기본 규칙**:
      - 당신의 주요 역할은 세법 정보를 사용자 친화적으로 전달하는 것입니다.
      - 데이터에 기반한 정보를 제공하며, 데이터에 없는 내용은 임의로 추측하지 않습니다.
      - 불확실한 경우, "잘 모르겠습니다."라고 명확히 답변하고, 사용자가 질문을 더 구체화하도록 유도합니다.

      **질문 처리 절차**:
      1. **질문의 핵심 내용 추출**:
          - 질문을 형태소 단위로 분석하여 조사를 무시하고 핵심 키워드만 추출합니다. 
          - 질문의 형태가 다르더라도 문맥의 의도가 같으면 동일한 질문으로 간주합니다.
          - 예를 들어, "개별소비세법 1조 알려줘" 와 "개별소비세법 1조는 뭐야" 와 "개별소비세법 1조의 내용은?"는 동일한 질문으로 간주합니다.
          - 예를 들어, "소득세는 무엇인가요?"와 "소득세가 무엇인가요?"는 동일한 질문으로 간주합니다.
      2. **관련 세법 조항 검색**:
          - 질문의 핵심 키워드와 가장 관련 있는 세법 조항이나 시행령을 우선적으로 찾습니다.
          - 필요한 경우, 질문과 연관된 추가 조항도 검토하여 답변의 완성도를 높입니다.
      3. **질문 유형 판단**:
          - **정의 질문**: 특정 용어나 제도의 정의를 묻는 경우.
          - **절차 질문**: 특정 제도의 적용이나 신고 방법을 묻는 경우.
          - **사례 질문**: 구체적인 상황에 대한 세법 해석을 요청하는 경우.
      4. **답변 생성**:
          - 법률 조항에 관한 질문이라면 그 조항에 관한 전체 내용을 가져온다.
          - 예를들어 '개별소비세법 1조의 내용'이라는 질문을 받으면 개별소비세법 1조의 조항을 전부 다 답변한다.
          - 질문 유형에 따라 관련 정보를 구조적으로 작성하며, 중요 세법 조문과 요약된 내용을 포함합니다.
          - 비전문가도 이해할 수 있도록 용어를 친절히 설명합니다.

      **답변 작성 가이드라인**:
      - **간결성**: 답변은 간단하고 명확하게 작성하되, 법 조항에 관한 질문일 경우 관련 법 조문의 전문을 명시합니다.
      - **구조화된 정보 제공**:
          - 세법 조항 번호, 세법 조항의 정의, 시행령, 관련 규정을 구체적으로 명시합니다.
          - 복잡한 개념은 예시를 들어 설명하거나, 단계적으로 안내합니다.
      - **신뢰성 강조**:
          - 답변이 법적 조언이 아니라 정보 제공 목적임을 명확히 알립니다.
          - "이 답변은 세법 관련 정보를 바탕으로 작성되었으며, 구체적인 상황에 따라 전문가의 추가 조언이 필요할 수 있습니다."를 추가합니다.
      - **정확성**:
          - 법령 및 법률에 관한질문은 추가적인 내용없이 한가지 content에 집중하여 답변한다.
          - 법조항에대한 질문은 시행령이나 시행규칙보단 해당법에서 가져오는것에 집중한다.

      **추가적인 사용자 지원**:
      - 답변 후 사용자에게 주제와 관련된 후속 질문 두 가지를 제안합니다.
      - 후속 질문은 사용자가 더 깊이 탐구할 수 있도록 설계하며, 각 질문 앞뒤에 한 줄씩 띄어쓰기를 합니다.

      **예외 상황 처리**:
      - 사용자가 질문을 모호하게 작성한 경우:
          - "질문이 명확하지 않습니다. 구체적으로 어떤 부분을 알고 싶으신지 말씀해 주시겠어요?"와 같은 문구로 추가 정보를 요청합니다.
      - 질문이 세법과 직접 관련이 없는 경우:
          - "이 질문은 제가 학습한 대한민국 세법 범위를 벗어납니다."라고 알리고, 세법과 관련된 새로운 질문을 유도합니다.

      **추가 지침**:
      - 개행문자 두 개 이상은 절대 사용하지 마세요.
      - 질문 및 답변에서 사용된 세법 조문은 최신 데이터에 기반해야 합니다.
      - 질문이 복합적인 경우, 각 하위 질문에 대해 별도로 답변하거나, 사용자에게 우선순위를 확인합니다.

      **예시 답변 템플릿**:
      - "질문에 대한 답변: ..."
      - "관련 세법 조항: ..."
      - "추가 설명: ..."

      {context}
      """),
      ("human", "{question}"),
  ]
  prompt_template = ChatPromptTemplate(messages)
  # 모델
  model = ChatOpenAI(model="gpt-4o")
  
  # output parser
  parser = StrOutputParser()
  
  # Chain 구성 retriever(관련문서 조회) -> prompt_template(prompt 생성) -> model(정답) -> output parser
  chain = {"context":retriever, "question": RunnablePassthrough()} | prompt_template | model | parser
  ```
  
---
## 📌 3. 성능 검증 및 평가
- **검증 방법:** 
- OPENAI를 이용한 실제 법령에 명시된 내용에서 Q-A쌍 10개 추출 후 LLM의 답변과 정확성, 관련성, 신뢰성 등 평가 지표 설정 및 측정
- 홈텍스 내의 질의응답 게시판과 LLM모델의 답변 비교
- cos 유사도를 통한 질문과 LLM답변간의 문백 유사도 검증
- bleu스코어 및 rouge 스코어 검증
- **결과:** 시스템이 제공하는 답변의 정확성과 일관성을 지속적으로 모니터링하여 시스템을 개선

```python
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

# 평가 데이터로 사용할 5개의 context 추출
total_samples = 5

idx_list = list(range(len(all_documents)))
random.shuffle(idx_list)

eval_context_list = []
while len(eval_context_list) < total_samples:
    idx = idx_list.pop()
    context = all_documents[idx].page_content
    if len(context) > 100:
        eval_context_list.append(context)

# 전체 context sample들로 qa dataset을 생성
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
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>context_recall</th>
      <th>llm_context_precision_with_reference</th>
      <th>faithfulness</th>
      <th>answer_relevancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>다국적기업그룹의 소득산입보완규칙 추가세액의 국내 배분액이 영으로 간주되는 경우는?</td>
      <td>[137 ❚상세본소득산입보완규칙의 구성기업별 배분방법 보완 (국조법 §73⑤) 11...</td>
      <td>대한민국 국조법 제73조의 규정에 따르면, 다국적기업그룹의 소득산입보완규칙 추가세액...</td>
      <td>적격소득산입보완규칙 추가세액배분액을 회계상 당기법인세비용으로 계상하지 않은 경우.</td>
      <td>NaN</td>
      <td>0.638889</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>최소적용제외 특례가 적용되는 요건은 무엇인가?</td>
      <td>[핵심 개정 세법 138최소적용제외 특례의 예외 신설 (국조법 §74②) 12 종 ...</td>
      <td>질문에 대한 답변: 최소적용제외 특례의 요건은 다음과 같습니다:\n1. 국가 내 구...</td>
      <td>해당 국가의 매출액과 글로벌최저한세소득ㆍ결손 금액이 각각 대통령령 기준 미만이어야 한다.</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.505570</td>
    </tr>
    <tr>
      <th>2</th>
      <td>미상각잔액 합계액을 계산하는 방법은 무엇인가?</td>
      <td>[귀속되는 이자등 상당액을 공제한 금 액으로 한다. ② 법 제87조의12제1항제2호...</td>
      <td>미상각잔액 합계액의 계산 방법은 관련 세법 및 조항에서 명시하고 있는 절차와 공식을...</td>
      <td>미상각잔액 합계액은 양도법인 및 양수법인이 계상한 해당 동종자산의 미상각잔액을 더한...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.360886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>적격합병등취득자산의 감가상각비 손금산입을 위한 취득가액 계산 방법은?</td>
      <td>[•• 취득하기 위하여 금융회사 등 또는 「주택도시기금법」에 따른 주택도시기금으로부...</td>
      <td>적격합병 등의 취득자산에 대한 감가상각비의 손금산입을 위해서는 해당 자산의 취득가액...</td>
      <td>적격합병등취득자산의 취득가액은 양도법인의 취득가액으로 하고, 미상각잔액은 양도법인의...</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>생산직 근로자의 야간수당은 비과세 소득인가요?</td>
      <td>[참고공장·광산근로자 중 야간근로수당 등이 비과세되지 아니하는 직종(예시) ∙구내이...</td>
      <td>생산직 근로자의 야간근로수당은 일정 요건을 충족하는 경우 비과세 소득으로 인정될 수...</td>
      <td>비과세 소득입니다.</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>주식매수선택권 관련 비과세 여부는 무엇인가요?</td>
      <td>[개별소비세법 시행령 기획재정부 (환경에너지세제과) 044-215-4331, 433...</td>
      <td>주식매수선택권과 관련된 비과세 여부는 다음과 같습니다:\n\n- **행사이익 비과세...</td>
      <td>비과세입니다.</td>
      <td>NaN</td>
      <td>0.500000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>신탁재산에서 발생하는 소득의 내용별 구분 예외는 언제부터 적용되나요?</td>
      <td>[핵심 개정 세법 26ㅇ(국조법 §58, §59, §91) 역외 세원관리 강화를 위...</td>
      <td>조각투자상품인 수익증권 발행신탁의 이익에 대한 소득 구분 원칙의 예외는 2025년 ...</td>
      <td>2025년 1월 1일 이후 개시하는 과세기간 분부터 적용됩니다.</td>
      <td>NaN</td>
      <td>0.887500</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>기업의 출산지원금 근로소득 비과세는 언제부터 적용되나요?</td>
      <td>[핵심 개정 세법 803 봉급 생활자 세금제도 (소득법) 기업의 출산지원금 근로소득...</td>
      <td>기업의 출산지원금 근로소득 비과세는 2024년 1월 1일 이후 지급받는 분부터 적용...</td>
      <td>2024년 1월 1일 이후 지급받는 분부터 적용됩니다.</td>
      <td>1.0</td>
      <td>0.700000</td>
      <td>1.000000</td>
      <td>0.378037</td>
    </tr>
    <tr>
      <th>8</th>
      <td>국내원천소득 중 원천징수세액이 1천원 미만인 소득은 무엇이 제외되는가?</td>
      <td>[Ⅰ. 사업소득 연말정산 339 09 사업소득 원천징수영수증(연말정산용) 작성박영영...</td>
      <td>국내원천소득 중 원천징수세액이 1천원 미만인 소득은 일반적으로에 원천징수가 면제되지...</td>
      <td>법 제119조제9호 및 제11호에 따른 소득은 제외된다.</td>
      <td>1.0</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>국내원천소득과 관련된 지급명세서 제출 의무는 어떤 법 조항에 따라 이루어지는가?</td>
      <td>[핵심 개정 세법 74국내원천 인적용역소득에 대한 비과세･면제신청서 및 지급명세서 ...</td>
      <td>국내원천소득과 관련된 지급명세서 제출 의무는 소득세법 제156조의2와 관련된 소득세...</td>
      <td>법 제46조 또는 제156조제6항ㆍ제16항에 따라 지급명세서를 제출해야 한다.</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

- 결과<br/>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team/blob/main/result_img/hometax.png"><br/>
<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team/blob/main/result_img/hometax_response.png"><br/>

답변: 
 취업 전 지출한 월세나 신용카드 결제 금액의 연말정산 공제 여부는 특정 조건에 따라 달라질 수 있습니다.일반적으로 연말정산에서 소득공제는 해당 과세기간 내 소득이 있는 경우에만 공제 가능합니다.따라서, 취업 이전에 소득이 없다면 해당 기간 동안의 지출에 대해서는 소득공제를 받을 수 있는 기준이 충족되지 않을 수 있습니다.
1. 월세 공제: 월세 세액공제는 근로소득이 있는 경우 해당 과세기간에 실제로 납부한 월세에 대해 적용됩니다. 따라서, 취업 전 월세 지출은 공제 대상이 아닐 가능성이 높습니다.
2. 신용카드 공제: 신용카드 사용금액 소득공제는 근로소득이 있는 거주자가 해당 과세연도에 사용한 금액에 대해 적용됩니다.취업 전 사용한 금액은 소득이 없었기 때문에 공제 대상이 아닐 수 있습니다.
각 개인의 상황에 따라 다를 수 있으므로, 정확한 판단을 위해서는 세무 전문가와 상담하거나 국세청의 지침을 확인하는 것이 좋습니다. 이 답변은 세법 관련 정보를 바탕으로 작성되었으며, 구체적인 상황에 따라 전문가의 추가 조언이 필요할 수 있습니다. <br/>
- 결과<br/>
Semantic Similarity: 0.38 <br/><br/>
BLEU점수:0.00 <br/><br/>
Rouge1점수:Score(precision=0.2, recall=0.5, fmeasure=0.28571428571428575) <br/>
RougeL점수:Score(precision=0.2, recall=0.5, fmeasure=0.28571428571428575) <br/>



---
## 📌 4. 한 줄 회고📝
 - **노원재**: 어렵군요..
 - **박서윤**: 이제까지 한 프로젝트 중에서 제일 재밌는데 잘 모르겠다,,,
 - **박유나**: 이번 프로젝트를 진행하며, 전처리 과정에 따라 결과가 크게 달라지는 점, prompt template 메시지 작성 방식에 따라 결과가 극명하게 변화하는 점이 흥미로웠습니다. 
 - **유경상**: 지난번 프로젝트때 부족했던 평가부분을 수정하면서 LLM모델 구성에대한 이해도가 높아졌다. 하지만 아직 장고활용에 있어서 부족한점이 많아 추가적으로 학습이 필요할거같다. 
 - **전하연**: 쉽지 않다..
>>>>>>> 71306b5 (챗봇)
