# 💸 Chatbot - 대한민국 세법 질의응답 챗봇 💸

## 📍목차
1. [팀원 소개](#1️⃣-팀원-소개)
2. [주제 소개](#2️⃣-주제-소개)
   - [주제](#주제)
   - [주제를 선택한 이유](#주제를-선택한-이유)
   - [주요 기능](#주요-기능)
3. [프로젝트 소개](#3️⃣-프로젝트-소개)
   - [디렉토리 구조](#디렉토리-구조)
   - [설치 및 실행 방법](#설치-및-실행-방법)
4. [RAG 모델 개선사항](#4️⃣-RAG-모델-개선사항)
   - [이전 모델의 문제점](#이전-모델의-문제점)
   - [개선 내용 정리](#개선-내용-정리)
5. [요구사항 정의서](#5️⃣-요구사항-정의서)
6. [화면 설계서](#6️⃣-화면-설계서)
7. [시스템 구성도](#7️⃣-시스템-구성도)
8. [테스트 계획서 및 테스트 결과 보고서](#8️⃣-테스트-계획서-및-테스트-결과-보고서)
   - [테스트 계획서](#테스트-계획서)
   - [테스트 결과 보고서](#테스트-결과-보고서)
   - [테스트 결과 정리](#테스트-결과-정리)
9. [📝한 줄 회고](#📝-한-줄-회고)

---

<br/>

## 1️⃣ 팀원 소개


| 이름  | 담당 업무  | 
|-----|------|
|박유나|프로젝트 관리, Django BE, FE, 화면설계, RAG 모델|
|유경상|Django BE (chat 관련), 데이터 처리 및 RAG 모델 개선, LLM 모델 테스트|
|노원재|Django BE (chat 관련), LLM 테스트, App 테스트|
|전하연|Django BE (user 관련), 요구사항 정의서|
|박서윤|Django BE (user 관련), 요구사항 정의서, 시스템구성도|
<br/>

## 2️⃣ 주제 소개

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
   - RAGAS 도입

<br/>

## 3️⃣ 프로젝트 소개

### 디렉토리 구조

```
├── data/                                   # PDF 데이터 (75개)
├── lawbot/                                 # 📍Django
│    ├── chat/                              # app: 챗봇
│    │    ├── template/chatbot.html
│    │    ├── vector_store                  # vector store
│    │    ├── ...
│    │    ├── llm.py                        # chain
│    │    ├── urls.py
│    │    └── views.py
│    ├── config/              
│    │    ├── settings.py
│    │    ├── urls.py 
│    │    └── views.py
│    ├── media/profile_pics/                # 프로필사진
│    ├── static/                            # 정적 파일
│    │    ├── css
│    │    ├── img
│    │    └── js
│    ├── templates/                         # html
│    │        ├── home.html
│    │        └── layout.html
│    └── users/                             # app: 회원
│         ├── template/
│         │   ├── profile.html
│         │   └── register.html
│         ├── forms.py                      # Form 관련
│         ├── models.py        
│         ├── urls.py
│         └── views.py
├── report/                                 # 산출물 및 readme.md, image
├── .gitignore            
├── loader.py                               # 모델 생성 함수 모듈   
├── main.ipynb                              # 모델 생성         
└── readme.md  
```


### 설치 및 실행 방법

- 로컬 실행
1. 저장소 클론:  
   ```bash
   git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-4th-04Team
   ```  
2. 가상환경 설정 및 패키지 설치:  
   ```bash
   cd lawbot
   pip install -r requirements.txt
   ```  
3. 서버 실행:  
   ```bash
   python manage.py runserver
   ```
<br/>

<h2 id="4️⃣-RAG-모델-개선사항">4️⃣ RAG 모델 개선사항</h2>

 <i>* 이전 프로젝트 → [Github Repository](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN06-3rd-4Team) </i>

### 이전 모델의 문제점
1. embedding vector의 차원 수 초과 -> chunk사이즈 및 overlap 수치 저하로 인한 문맥 파악 능력 저하
2. 세법 데이터를 각 조항별로 split 하였음 → 컨텍스트 부족으로 인한 부정확한 응답 발생.
   <br/>
   - 예) "개별소비세법 <u>제1조</u>를 알려줘!" 라는 질문에 응답하기 어려워 함. 할루시네이션 발견.
   <br/> 
   → 각 세법에 따라 법률, 시행령, 시행규칙을 잘 구분하지 못하기 때문이라고 판단.
      - 개별소비세법 <u>법률 제1조</u> 
      - 개별소비세법 <u>시행령 제1조</u>
      - 개별소비세법 <u>시행규칙 제1조</u>
   

   - ∵ prompt 로 먼저 해결해 보고자 하였으나 평가점수가 역시나 낮았음. 
   - ∴ chunk 기반으로 split하여 문맥을 읽히기로 함.
   - 해결책:
      - 72개의 파일을 7개의 폴더로 나눈 후 각 폴더별 embedding 후 빈 vector store에 차례대로 추가 및 저장.
   - 결과: 매우 성공적!👌

3. RAGAS 평가 결과 미흡
   - precision, recall, relevency 확인 시 정답내용을 임의적으로 선별함.
   - 평가 데이터 부족, 몇가지 조항들로만 LLM모델 평가
 
   - 해결책:
      - 평가용 chain을 구성 -> 평가 데이터로 사용할 context를 추출 -> 평가데이터셋 구성 -> RAGAS 평가
      - 홈텍스의 Q&A 와 LLM모델의 응답 비교
      - LLM모델의 응답과 홈텍스의 Q&A 정답간의 문맥 유사도 비교
      - LLM모델의 응답과 홈텍스의 Q&A 정답간의 BLEU, ROUGE Score 를 통한 평가
   - 결과: 매우 성공적!👌
   - LLM 테스트 보고서 → [Chatbot README.md](/lawbot/chat/README.md)


### 개선 내용 정리

1. **벡터스토어 최적화**
   - Chroma Vectorstore의 효율적인 인덱싱 및 쿼리 처리 개선.

2. **Chunk 기반 데이터 스플릿**
   - 데이터를 문맥 단위로 분리하여 질의응답 정확성 향상.
   - 응답의 일관성과 신뢰도 15% 이상 증가.

3. **RAGAS 의 확실한 도입**
   - 응답 속도 최적화 및 사용자의 요구에 따른 적응형 모델 구조.
   - Q-A질문 쌍이 주어진 context에 얼마나 부합한지 재현율, 정밀도, 신뢰성, 적합성

4. **다양한 평가지표 도입**
   - RAGAS를 통한 context
   - 홈텍스 내의 질의응답 게시판과 LLM모델의 답변 비교
   - cos 유사도를 통한 질문과 LLM답변간의 문백 유사도 검증
   - bleu스코어 및 rouge 스코어 검증

## 5️⃣ 요구사항 정의서

📍 <i>자세한 <u>**&lt;요구사항 정의서&gt;**</u>  문서 확인 → [Google sheet](https://docs.google.com/spreadsheets/d/1wKG5qj-1ep-ace8Mv8cm0a9_KxEt96GTHUgf0Eu_epk/edit?gid=0#gid=0) 참고 </i>

![](</report/요구사항_정의서.png>)

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

## 6️⃣ 화면 설계서

📍 <i>자세한 <u>**&lt;화면 설계서&gt;**</u>  문서 확인 → [Figma](https://www.figma.com/design/lHj5Rco1lt6BvE4YdrM6ys/SKN06_4th_T4_%ED%99%94%EB%A9%B4%EC%84%A4%EA%B3%84%EC%84%9C?node-id=0-1&t=iVbEYIeSSG7Xd9VC-1) 참고 </i>

![](</report/화면설계서.png>)
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

## 7️⃣ 시스템 구성도
![](</report/시스템_구성도.png>)


<br/>

## 8️⃣ 테스트 계획서 및 테스트 결과 보고서

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
![](</report/테스트_계획서_및_테스트_결과_보고서.png>)

 - 기능 테스트 결과

   | 기능                | 테스트 항목                     | 예상 결과                         | 실제 결과             | 상태  |
   |---------------------|--------------------------------|----------------------------------|-----------------------|-------|
   | **홈페이지 접속**   | 접속 화면                      | 홈 화면 표시                      | 로그인 화면 표시       | **Pass** |
   | **회원가입**        | 필수 입력 완료                 | 채팅 화면으로 이동                | 채팅 화면 이동         | **Pass** |
   |                     | 입력 사항 미완료               | 오류 메시지 표시                  | 오류 메시지 표시       | **Pass** |
   |                     | 이메일 형식 오류               | 오류 메시지 표시                  | 오류 메시지 표시       | **Pass** |
   |                     | 비밀번호 불일치                | 오류 메시지 표시                  | 오류 메시지 표시       | **Pass** |
   | **로그인**          | 성공                           | 채팅 화면으로 이동                | 채팅 화면 이동         | **Pass** |
   |                     | 실패                           | 오류 메시지 표시                  | 오류 메시지 표시       | **Pass** |
   | **로그아웃**        | 로그아웃 버튼 클릭             | 로그인 화면 표시                  | 로그인 화면 표시       | **Pass** |
   | **마이페이지**      | 프로필 수정                    | 수정 사항 반영                    | 정보 수정 완료         | **Pass** |
   |                     | 비밀번호 변경 실패             | 오류 메시지 표시                  | 오류 메시지 표시       | **Pass** |

<br/>
 
 - 챗봇 테스트 결과

   | 테스트 항목              | 예상 결과                    | 실제 결과               | 상태  |
   |--------------------------|-----------------------------|-------------------------|-------|
   | 메시지 전송             | 관련 법령 정보 제공         | 관련 법령 정보 제공      | **Pass** |
   | 세션 유지 실패          | 이전 대화 유지              | 새로운 대화 시작         | **Fail** |
   | 카테고리 클릭           | 관련 안내 메시지 제공       | 관련 안내 메시지 제공    | **Pass** |
   | 입력 필드 비활성화      | 입력창 비활성화             | 입력창 비활성화          | **Pass** |
   | 대화 초기화             | 새로운 대화 시작            | 새로운 대화 시작         | **Pass** |


### 테스트 결과 정리
   - **강점**:
      - 사용자 입력 검증(회원가입, 로그인)의 정확성.
      - 챗봇의 정보 제공 기능 및 대화 초기화 기능 안정성.

   2. **개선 필요점**:
      - **세션 유지 문제**:
      - 새로고침 후 이전 대화가 유지되지 않음.
      - 쿠키, DB 저장 등의 방법을 고려해 볼 수 있을 것 같음.
      - 사용자 경험을 개선하기 위해 세션 복원 기능 강화 필요.

<br/>

## 📝 한 줄 회고
 - **노원재**: 지금까지 배운 내용을 실제로 시각화하여 적용하고 실시간으로 변동사항이 반영되는 것을 확인할 수 있어서 흥미로웠다.

 - **박서윤**: 실제로 만들어보면서 장고 구성에 대해서 파악할 수 있어서 좋았던 것 같습니다.

 - **박유나**: 파이썬으로 웹 애플리케이션을 구현해본 것은 처음이었는데, 매우 흥미로운 경험이었다. 특히 AI 모델을 활용해 웹상에서 서비스를 구현해볼 수 있었던 점이 인상 깊었다.

 - **유경상**: 지난번에 미흡했던 LLM부분에 대해 더 알게되었고 성능이 좋아져서 다행이었다. 모델 평가부분 또한 몰랐던 부분들을 배우며 적용시켰고 장고에서 챗봇을 기존 모델에서 가져오는방법과 구현 로직에대해 좀 더 이해할 수 있는 기회여서 좋았다.
 
 - **전하연**: 코드를 수정하면서 결과를 바로바로 확인할 수 있어 장고에 대해 알아가는 재미가 있었습니다.
