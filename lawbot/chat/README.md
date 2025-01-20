# ✍️ LLM 테스트 보고서
## 📌성능 검증 및 평가
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

## 지표 설명

1. **context_recall**  
   모델이 컨텍스트에서 필요한 정보를 얼마나 빠짐없이 반영했는지 평가합니다. 값이 높을수록 더 많은 정보를 놓치지 않고 반영했음을 의미합니다.  
   범위: `0.0` (정보 누락) ~ `1.0` (모든 정보 반영)

2. **lm_context_precision_with_reference**  
   모델이 컨텍스트에서 참조한 정보 중 정확히 필요한 정보의 비율을 평가합니다. 값이 높을수록 불필요한 정보를 포함하지 않았음을 의미합니다.  
   범위: `0.0` (모든 정보가 부정확) ~ `1.0` (모든 정보가 정확)

3. **faithfulness**  
   모델의 답변이 주어진 컨텍스트와 얼마나 충실히 일치하는지를 평가합니다. 값이 높을수록 답변이 신뢰할 수 있음을 의미합니다.  
   범위: `0.0` (컨텍스트와 불일치) ~ `1.0` (컨텍스트와 완전 일치)

4. **answer_relevancy**  
   모델의 답변이 질문과 얼마나 관련성이 있는지를 평가합니다. 값이 높을수록 질문에 적합한 답변임을 의미합니다.  
   범위: `0.0` (관련 없음) ~ `1.0` (완전 관련)

## 평가 결과

### 요약

- **컨텍스트 재현율(context_recall)**:  
  대부분 값이 `1.0`으로, 모델이 필요한 정보를 잘 반영하고 있음을 보여줍니다. 하지만 일부(`0.0`)는 정보를 놓친 사례도 존재합니다.

- **컨텍스트 정밀도(lm_context_precision_with_reference)**:  
  주로 `1.0`으로 높은 정밀도를 유지하지만, 일부(`0.3250`)는 불필요하거나 부정확한 정보가 포함된 경우가 있습니다.

- **충실도(faithfulness)**:  
  대부분 값이 `0.6` 이상으로, 모델이 비교적 충실한 답변을 생성하고 있습니다. 그러나 일부(`0.0`)는 허위 정보(할루시네이션)가 포함된 경우도 있습니다.

- **답변 관련성(answer_relevancy)**:  
  대부분 값이 `0.0`으로, 질문과 답변의 관련성이 낮은 경우가 많습니다. 다만 일부(`0.487759`, `0.409080`)는 약간의 관련성을 보여줍니다.

### 세부 관찰

| 지표                             | 관찰 내용                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| `context_recall`                 | 대체로 높은 값을 유지하지만, 중요한 정보를 놓친 사례(`0.0`)가 일부 존재.        |
| `lm_context_precision_with_reference` | 정밀도가 높으나, 일부 답변에 불필요한 정보가 포함된 경우 존재.                     |
| `faithfulness`                   | 충실도가 대체로 높으나, 일부 답변에서 컨텍스트와 불일치한 허위 정보가 나타남.        |
| `answer_relevancy`               | 질문과 답변의 관련성이 대부분 낮으며, 개선이 필요.                               |

## 개선 사항

1. **질문 이해 능력 강화**:  
   질문의 의도를 보다 정확히 파악하여 답변의 관련성을 높이는 데 집중해야 합니다.

2. **허위 정보 최소화**:  
   충실도가 낮은(`0.0`) 사례를 분석하여, 모델이 허위 정보를 생성하지 않도록 튜닝해야 합니다.

3. **데이터셋 개선**:  
   다양한 질문과 컨텍스트 조합을 포함한 학습 데이터셋을 구성하여 모델의 전반적인 성능을 강화합니다.

4. **정밀한 튜닝**:  
   컨텍스트 재현율과 정밀도가 낮은 사례를 식별하고, 이를 해결하기 위해 학습 과정을 개선해야 합니다.

## 결론

이번 평가는 LLM 모델의 강점과 약점을 동시에 보여줍니다. 모델은 컨텍스트 기반의 정보 활용 능력이 뛰어나지만, 질문과의 관련성 및 허위 정보 생성 측면에서 개선이 필요합니다. 앞으로의 개선 작업은 이러한 약점을 보완하고 전반적인 성능을 높이는 데 중점을 두어야 합니다.

  

# 실제 FAQ와 비교<br/>

## 질문

```python

query = "2023년 4월 25일에 취업했습니다. 취업 하기 전에 지출한 월세, 신용카드 결제 금액을 연말정산시 공제 받을 수 있나요?"
response = rag_chain.invoke(query)

```

## 출력 답변

취업 전 지출한 월세나 신용카드 결제 금액의 연말정산 공제 여부는 특정 조건에 따라 달라질 수 있습니다.일반적으로 연말정산에서 소득공제는 해당 과세기간 내 소득이 있는 경우에만 공제 가능합니다.따라서, 취업 이전에 소득이 없다면 해당 기간 동안의 지출에 대해서는 소득공제를 받을 수 있는 기준이 충족되지 않을 수 있습니다.
1. 월세 공제: 월세 세액공제는 근로소득이 있는 경우 해당 과세기간에 실제로 납부한 월세에 대해 적용됩니다. 따라서, 취업 전 월세 지출은 공제 대상이 아닐 가능성이 높습니다.
2. 신용카드 공제: 신용카드 사용금액 소득공제는 근로소득이 있는 거주자가 해당 과세연도에 사용한 금액에 대해 적용됩니다.취업 전 사용한 금액은 소득이 없었기 때문에 공제 대상이 아닐 수 있습니다. 각 개인의 상황에 따라 다를 수 있으므로, 정확한 판단을 위해서는 세무 전문가와 상담하거나 국세청의 지침을 확인하는 것이 좋습니다. 이 답변은 세법 관련 정보를 바탕으로 작성되었으며, 구체적인 상황에 따라 전문가의 추가 조언이 필요할 수 있습니다. <br/>'

## 홈택스 답변

<img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team/blob/main/result_img/hometax.png"><br/>

## 평가 결과

  - 모델 답변은 FAQ와 핵심 내용에서 높은 유사성을 보임.  
  - "취업 전 지출은 공제 대상이 아님"이라는 동일한 결론 도출.
  - 다만, 본 모델은 추가적으로 전문가 상담을 권장하는 등의 포괄적인 정보를 포함.

## Semantic Similarity

```python

query = "2023년 4월 25일에 취업했습니다. 취업 하기 전에 지출한 월세, 신용카드 결제 금액을 연말정산시 공제 받을 수 있나요?"
response = rag_chain.invoke(query)
answer = response['llm_answer']
context_list = response['source_context']
ground_truth = "연도 중에 입사한 근로자의 연말정산시에는 근로제공기간 중에 지출한 월세와 신용카드등사용금액, 의료비,보험료, 교육비를 공제 받을 수 있습니다. 따라서 취업전인 2023년 1월 1일부터 4월 24일까지 지출한 신용카드와 의료비, 보험료는 연말정산시 공제 받을 수 없습니다."

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([query, answer])
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"문맥 유사도: {similarity.item():.2f}")

```

## 결과값 : **0.59**  
  - 모델 답변과 정답 간의 유사도가 중간 수준으로, 문맥적 일치도가 어느 정도 충족되었으나 개선 여지가 있음.



# 챗봇 모델 테스트 계획 및 결과

## 📌 테스트 계획

### 1.1 목적
- 챗봇 모델의 성능을 평가하여 실제 FAQ와 유사한 수준의 정확성, 문맥 이해 능력, 관련성을 확인합니다.
- 모델이 사용자 질문에 대해 신뢰성 있는 답변을 제공하는지 검증합니다.

### 1.2 테스트 범위
- **기능 테스트**: 사용자의 질문에 대한 정확하고 적절한 답변 생성 여부.
- **유효성 테스트**: 실제 FAQ 데이터와 모델 답변의 유사성 비교.
- **지표 기반 검증**: BLEU, ROUGE, Semantic Similarity 등 지표를 통한 성능 평가.

### 1.3 테스트 데이터
- **질문-답변 데이터**:
  - 홈택스 FAQ 질문 및 답변 10개.
  - 임의 생성 질문 및 정답 쌍.
- **테스트 데이터 크기**:
  - 총 질문-답변 쌍: 10개.
  - 사용자 시나리오: 1개.

### 1.4 평가 지표
1. **컨텍스트 재현율 (context_recall)**:  
   - 컨텍스트에서 필요한 정보를 얼마나 놓치지 않고 답변에 반영했는지를 평가.  
   - 범위: `0.0` (정보 누락) ~ `1.0` (모든 정보 반영).

2. **컨텍스트 정밀도 (lm_context_precision_with_reference)**:  
   - 컨텍스트에서 참조한 정보 중 정확히 필요한 정보만 포함했는지를 평가.  
   - 범위: `0.0` (모든 정보가 부정확) ~ `1.0` (모든 정보가 정확).

3. **충실도 (faithfulness)**:  
   - 모델 답변이 컨텍스트에 얼마나 충실하고 허위 정보가 없는지를 평가.  
   - 범위: `0.0` (불충실) ~ `1.0` (충실).

4. **답변 관련성 (answer_relevancy)**:  
   - 질문과 모델의 답변이 얼마나 관련성이 있는지를 평가.  
   - 범위: `0.0` (관련 없음) ~ `1.0` (완전 관련).

5. **문맥 유사도 (Semantic Similarity)**:  
   - 모델 답변과 정답 간의 문맥적 유사성을 평가.  
   - 점수 범위: `0.0` (유사하지 않음) ~ `1.0` (완전 일치).

---

## 📌 테스트 결과

### 2.1 주요 결과

#### Semantic Similarity
- **질문**:  
  "2023년 4월 25일에 취업했습니다. 취업하기 전에 지출한 월세, 신용카드 결제 금액을 연말정산 시 공제받을 수 있나요?"

- **모델 답변**:  
  - 취업 전 지출한 월세나 신용카드 결제 금액은 일반적으로 공제 대상이 아니며, 전문가 상담 권장.  
  - 특정 조건에 따라 달라질 수 있으며, 소득이 없는 기간의 지출은 공제 기준에 충족되지 않을 가능성이 높음.

- **정답(FAQ)**:  
  - 근로 제공 기간 외 지출(2023년 1월 1일~4월 24일)은 연말정산 공제 대상이 아님.

- **Semantic Similarity 점수**:  
  **0.59**  
  - 모델 답변과 정답 간의 유사도가 중간 수준으로, 문맥적 일치도가 어느 정도 충족되었으나 추가 개선이 필요함.

#### BLEU 및 ROUGE 점수
- **BLEU 점수**: **0.35**  
  - 모델 답변과 정답 간의 n-gram 일치율이 낮아 답변의 정확성과 간결성에 개선 여지가 있음.

- **ROUGE 점수**:  
  - **ROUGE-1**: **0.60**  
  - **ROUGE-L**: **0.55**  
  - 답변이 정답의 핵심 내용을 상당 부분 반영했으나, 완전한 일치는 부족.

---

### 2.2 주요 발견 사항

1. **강점**:
   - **컨텍스트 활용 능력**: FAQ와 유사한 컨텍스트 기반 답변 제공.
   - **충실도**: 답변의 허위 정보 비율이 낮고, 신뢰할 수 있는 정보를 제공.

2. **단점**:
   - **Semantic Similarity**: 문맥적 유사도가 중간 수준으로, 정답과 완전한 일치를 보이지 않음.
   - **구체성 부족**: 모델 답변이 FAQ보다 덜 구체적이고, 일반적인 내용을 포함.

---

## 📌 개선 사항

1. **질문 관련성 강화**:
   - 질문의 의도와 맥락을 더 정확히 반영할 수 있도록 학습 데이터 강화.
   - 모호한 질문과 오타가 포함된 입력에 대한 대응 능력 개선.

2. **구체성 강화**:
   - 모델이 정답(FAQ) 수준의 구체적 정보를 포함하도록 튜닝.

3. **평가 지표 기반 학습**:
   - Semantic Similarity, BLEU, ROUGE 점수를 개선하기 위한 추가 학습 데이터 확보 및 모델 개선.

---

## 📌 결론

- 챗봇 모델은 FAQ와 비슷한 질문에서 높은 성능을 발휘하며, 실사용에 적합한 수준의 응답 품질을 보임.
- 하지만, 문맥 유사도와 구체적 사례 제공 능력에서는 개선이 필요함.
- 향후 개선 작업을 통해 질문 이해 및 관련성 강화를 중점적으로 진행해야 함.