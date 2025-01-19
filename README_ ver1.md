
<!-- SKN06-4TH-04Team


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
 -->

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

## 📌 RAG 기반 모델 

### 데이터 수집
-  [법제처](https://www.law.go.kr/LSW/main.html)와 [국세청](https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2304&cntntsId=238938)에서 세법 관련 데이터 다운로드

### 데이터 로드
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
### 데이터 전처리
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
    
    
### split 방법
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



### 벡터 데이터베이스 구현 
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
  
---
## 📌 성능 검증 및 평가
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
