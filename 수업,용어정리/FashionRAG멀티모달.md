# FashionRAG: 멀티모달 RAG 수업 정리

> CLIP 이미지 검색과 GPT 비전 모델로 만드는 패션 스타일링 어시스턴트
> 결과물: 한국어로 질문하면 비슷한 패션 사진을 찾아 보여 주고, 그 사진을 근거로 스타일링 조언을 해 주는 챗봇

---

## 1. 핵심 개념

### RAG (Retrieval-Augmented Generation)

LLM은 학습하지 않은 정보(우리 회사 상품, 최신 데이터 등)를 모른다. 그래서 세 단계로 답하게 한다.

1. **Retrieval (검색):** 질문과 관련된 자료를 먼저 찾는다.
2. **Augmentation (보강):** 찾은 자료를 프롬프트에 붙여 넣는다.
3. **Generation (생성):** LLM이 그 자료를 근거로 답한다.

### 멀티모달 RAG

- **모달리티(modality)** 는 데이터의 종류를 말한다. 텍스트, 이미지, 음성, 영상 등이 있다.
- 일반 RAG는 텍스트 문서를 검색한다. 멀티모달 RAG는 **이미지를 검색**하고, 이미지를 볼 수 있는 LLM(**비전 모델**)에게 넘겨서 답하게 한다.

### 임베딩 (Embedding)

- 데이터를 **의미를 담은 숫자 벡터**로 바꾼 것이다.
- 의미가 비슷하면 벡터도 서로 가깝다.

### CLIP (Contrastive Language-Image Pre-training)

- OpenAI가 2021년에 발표한 구조다. 수업에서는 LAION이 공개 데이터(약 20억 쌍)로 학습한 **OpenCLIP ViT-B-32** 를 썼다.
- **이미지 인코더**와 **텍스트 인코더** 두 개로 이루어져 있고, 둘 다 **512차원** 벡터를 출력한다.
- **대조 학습:** "사진–설명" 짝은 벡터를 가깝게, 짝이 아닌 것은 멀게 학습한다.
- 그 결과 **텍스트와 이미지가 같은 벡터 공간**에 놓인다. 그래서 `"a red dress"` 라는 텍스트로 빨간 드레스 사진을 찾을 수 있다.
- **한계:** 주로 영어로 학습돼서, 한국어 검색어("빨간 드레스")를 넣으면 결과가 부정확하다. 그래서 FashionRAG는 질문을 먼저 영어로 번역한다.

### 거리 (distance) 읽는 법

- Chroma의 기본 거리는 **제곱 유클리드 거리(L2²)** 이고, **작을수록 비슷하다**.
- CLIP 벡터를 길이 1로 정규화하면 `L2² = 2 − 2 × 코사인유사도` 가 성립한다. 그래서 순위는 코사인 유사도 순위와 같다.
- 헷갈리지 않게 정리하면 이렇다.
  - **코사인 유사도:** 1에 가까울수록 비슷하다.
  - **Chroma distance:** 0에 가까울수록 비슷하다.

---

## 2. 전체 흐름

```
 사용자(한국어): "가을 트렌치코트 스타일링 알려줘"
        │
        ▼  ① 번역 체인 (gpt-4o-mini)                    전처리
 "Trench coat styling for autumn"
        │
        ▼  ② CLIP 텍스트 임베딩 → Chroma 유사도 검색    Retrieval
 [image_0123.jpg, image_0456.jpg]  (가장 비슷한 사진 2장)
        │
        ▼  ③ 이미지 → base64 data URL, 질문과 합쳐 프롬프트 구성   Augmentation
        │
        ▼  ④ 비전 체인 (gpt-4o-mini가 이미지를 보고 답변)          Generation
 "In the first image, the beige trench coat ..."
        │
        ▼  ⑤ 번역 체인 (영어 → 한국어)                  후처리
 사용자에게 한국어 답변과 검색된 사진을 보여 줌
```

**사전 준비 (처음 한 번만):** Fashionpedia 데이터셋에서 사진을 `fashion_dataset/` 폴더에 저장한 뒤, CLIP으로 임베딩해 Chroma 벡터 DB(`img_vdb/`)에 넣는다.

| 단계 | 함수 | RAG 역할 |
|---|---|---|
| ① 질문 번역 | `translate(query_ko, "English")` | 전처리 |
| ② 이미지 검색 | `query_db(image_vdb, query_en)` | **R**etrieval |
| ③ 프롬프트 입력 구성 | `format_prompt_inputs(results, query_en)` | **A**ugmentation |
| ④ 답변 생성 | `vision_chain.invoke(prompt_input)` | **G**eneration |
| ⑤ 답변 번역 | `translate(response_en, "Korean")` | 후처리 |

---

## 3. 사용한 라이브러리

| 라이브러리 | 역할 | 사용한 기능 |
|---|---|---|
| `datasets` | Hugging Face 데이터셋 로드 | `load_dataset(..., streaming=True)` |
| `open-clip-torch` | CLIP 모델 실행 | Chroma의 `OpenCLIPEmbeddingFunction` 내부에서 사용 |
| `chromadb` | 벡터 DB | `PersistentClient`, `add`, `query` |
| `langchain-core` | LLM 앱 구성요소 | `ChatPromptTemplate`, `StrOutputParser`, `\|` |
| `langchain-openai` | OpenAI 연결 | `ChatOpenAI` |
| `pillow` | 이미지 열기, 저장, 합치기 | `Image.open`, `thumbnail`, `paste` |
| `python-dotenv` | `.env` 에서 API 키 읽기 | `load_dotenv()` |

```python
%pip -q install chromadb open-clip-torch pillow datasets langchain-openai langchain-core python-dotenv
```

- 노트북에서는 `!pip` 보다 `%pip` 를 쓴다. `%pip` 는 지금 실행 중인 커널의 파이썬 환경에 설치하고, `!pip` 는 다른 환경에 설치될 수 있다.
- 설치 후 커널 재시작 안내가 나오면 재시작하고, 설치 셀은 건너뛴다.

---

## 4. 단계별 정리

### Part 1. 환경 준비

**API 키는 코드에 직접 쓰지 않는다.**

```python
load_dotenv()                                  # .env 파일을 환경변수로 등록
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass("OpenAI API 키를 입력하세요: ").strip()  # 입력이 화면에 안 보임
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # ChatOpenAI가 자동으로 읽음
print(OPENAI_API_KEY[:6] + "...")              # 확인할 때도 앞 6자리만 출력
```

**설정값은 한 셀에 모은다.** 경로, 모델명, 개수 같은 값을 맨 위에서만 고치면 전체 동작이 바뀌게 한다.

```python
BASE_DIR = os.getcwd()   # 노트북에는 __file__ 이 없어서 현재 작업 폴더를 기준으로 함
DATASET_FOLDER = os.path.join(BASE_DIR, "fashion_dataset")
VDB_PATH = os.path.join(BASE_DIR, "img_vdb")
COLLECTION_NAME = "fashion_images"
NUM_IMAGES = 100         # 가져올 이미지 수
N_RESULTS = 2            # 질문당 검색할 이미지 수
TRANSLATION_MODEL = VISION_MODEL = "gpt-4o-mini"
```

**보조 함수**

- `list_image_files(folder)`: 이미지 파일 목록을 정렬해서 반환한다.
- `show_image_grid(paths)`: 흰 캔버스를 만들고, 너비는 썸네일 너비의 합, 높이는 가장 큰 썸네일 높이로 정한 뒤 `paste` 로 이어 붙인다. matplotlib 없이 PIL만 쓴다.
- `guess_mime(path)`: 확장자로 MIME 타입을 정한다. `.png` 는 `image/png`, 나머지는 `image/jpeg` 다.

### Part 2. 데이터 준비 (Fashionpedia)

- Hugging Face ID는 `detection-datasets/fashionpedia` 다. train 45,623장, 27개 의류 카테고리가 있고 라이선스는 CC BY 4.0이다.
- 원래는 객체 탐지용 데이터셋이지만, 수업에서는 **사진만** 꺼내 검색 대상으로 썼다.
- Hugging Face에는 다양한 분야의 데이터셋이 있어서 바로 가져다 쓸 수 있다.

**스트리밍 모드**

| 방식 | 동작 | 500장만 쓸 때 |
|---|---|---|
| `load_dataset(id)` | 전체를 디스크에 받은 뒤 로드 | 약 3.5GB 전부 다운로드 |
| `load_dataset(id, streaming=True)` | 필요한 만큼만 순서대로 읽음 | 필요한 분량만 다운로드 |

- 스트리밍으로 받으면 `IterableDataset` 이 된다. `ds[0]` 같은 인덱스 접근은 안 되고, `for` 문이나 `.take(n)` 으로 꺼내 쓴다.

```python
for i, example in enumerate(dataset.take(num_images)):
    file_path = os.path.join(dataset_folder, f"image_{i+1:04d}.jpg")
    example["image"].convert("RGB").save(file_path, format="JPEG", quality=90)
```

- **`:04d` 로 0을 채운다** (`image_0001.jpg`). 이렇게 하면 문자열 정렬 순서와 번호 순서가 같아진다. 0을 채우지 않으면 `image_1, image_10, image_100, image_2...` 순서로 정렬된다.
- **JPEG로 저장한다.** PNG보다 용량이 작아서 나중에 base64로 보낼 때 데이터량도 줄어든다.
- **이미 저장돼 있으면 건너뛴다.** 파일이 `NUM_IMAGES` 장 이상 있으면 다운로드하지 않는다.

### Part 3. 임베딩과 벡터 DB (CLIP + Chroma)

| 개념 | 설명 | 코드 |
|---|---|---|
| Client | DB 연결. `PersistentClient` 는 디스크에 저장해서 노트북을 꺼도 남는다 (`Client()` 는 메모리 전용) | `chromadb.PersistentClient(path=...)` |
| Collection | 테이블 같은 단위 | `get_or_create_collection(name=...)` |
| Embedding Function | 데이터를 벡터로 자동 변환 | `OpenCLIPEmbeddingFunction()` |
| Data Loader | 경로(URI)를 실제 이미지로 읽음 | `ImageLoader()` |
| add / query | 저장 / 유사도 검색 | `add(ids, uris)`, `query(query_texts, n_results)` |

```python
chroma_client = chromadb.PersistentClient(path=VDB_PATH)
image_vdb = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=OpenCLIPEmbeddingFunction(),  # 처음 실행할 때 약 600MB 가중치 다운로드
    data_loader=ImageLoader(),
)
```

- **`add(uris=...)` 내부 동작:** 경로를 받으면 ImageLoader가 이미지를 읽고, CLIP 이미지 인코더가 벡터로 바꿔 저장한다.
- **`query(query_texts=...)` 내부 동작:** 텍스트를 CLIP 텍스트 인코더로 바꾼 뒤, 저장된 이미지 벡터들과의 거리를 계산해 가까운 순으로 돌려준다.

**원본 코드에서 고친 점**

- id를 파일명(`image_0001`)으로 정해서 id와 파일이 항상 일치하게 했다.
- DB에 이미 있는 id는 건너뛰고 새 이미지만 추가한다. 이렇게 하면 여러 번 실행해도 결과가 같다(**멱등성**).
- `batch_size=50` 씩 나눠서 넣는다. 진행 상황을 볼 수 있고 메모리 사용도 줄어든다.

```python
existing_ids = set(image_vdb.get(include=[])["ids"])   # include=[] 로 id만 가볍게 조회
```

**검색 결과의 구조**

검색어 1개, 결과 2개일 때 결과는 2차원 리스트로 온다.

```python
{'ids':       [['image_0042', 'image_0310']],
 'uris':      [['.../image_0042.jpg', '.../image_0310.jpg']],
 'distances': [[1.52, 1.55]]}
```

- 바깥 리스트는 검색어 순서, 안쪽 리스트는 순위 순서다. 그래서 `[0]` 으로 첫 번째 검색어의 결과를 꺼낸다.
- 실습 결과: `"orange one"` 으로 검색하니 1위 `image_0098` (distance 0.7808)이 나왔다. 이 단계는 OpenAI 없이 CLIP과 Chroma만으로 동작한다.

### Part 4. LangChain 체인

**LCEL (LangChain Expression Language)**

`|` (파이프) 연산자로 구성요소를 이어 체인을 만든다. 앞 단계의 출력이 다음 단계의 입력이 된다.

```
dict ──ChatPromptTemplate──▶ [SystemMessage, HumanMessage]
     ──ChatOpenAI──────────▶ AIMessage(content="...")
     ──StrOutputParser─────▶ "..." (str)
```

| 구성요소 | 역할 |
|---|---|
| `ChatPromptTemplate.from_messages([...])` | `("system", ...)`, `("user", ...)` 처럼 역할별 메시지 템플릿을 만든다. `{변수}` 자리표시자를 쓴다. |
| `ChatOpenAI(model, temperature)` | `temperature=0` 이면 매번 같은 답에 가깝고, 높을수록 다양해진다. 번역에는 0을 쓴다. |
| `StrOutputParser()` | `AIMessage` 에서 문자열만 꺼낸다. |
| `chain.invoke(dict)` | 체인을 실행한다. |

**번역 체인: 한 번만 만들고 재사용한다.**

```python
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate the following text to {target_lang}. "
               "Keep markdown formatting. Output only the translation."),
    ("user", "{text}"),
])
translation_chain = translation_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0) | StrOutputParser()

def translate(text, target_lang):
    return translation_chain.invoke({"text": text, "target_lang": target_lang})

translate("데이트할 때 입을 빨간 원피스", "English")  # → "Red dress to wear on a date"
```

**비전 모델에 이미지를 보내는 방법**

user 메시지를 **콘텐츠 블록 리스트**로 보낸다.

```python
("user", [
    {"type": "text", "text": "What are some ideas for styling {user_query}"},
    {"type": "image_url", "image_url": {"url": "data:{image_mime_1};base64,{image_data_1}"}},
    {"type": "image_url", "image_url": {"url": "data:{image_mime_2};base64,{image_data_2}"}},
])
```

**base64와 data URL**

- 이미지 파일은 바이너리라서 JSON 텍스트에 그대로 넣을 수 없다.
- **base64** 는 바이너리를 64개 문자(`A-Z a-z 0-9 + /`)로 바꾸는 인코딩이다. 용량은 약 1.33배로 늘어난다.
- **data URL** 은 `data:<MIME>;base64,<문자열>` 형태다. 파일을 인터넷에 올리지 않고 URL 자리에 직접 넣을 수 있다.
- MIME 타입은 실제 파일 형식과 맞춰야 한다. 원본 코드는 PNG 파일을 `image/jpeg` 로 고정해서 보냈는데, 이걸 변수로 바꿨다.

```python
def encode_image(path):
    with open(path, "rb") as f:                         # rb = 바이너리 읽기
        return base64.b64encode(f.read()).decode("utf-8")  # bytes → base64 bytes → str
```

- `format_prompt_inputs()` 는 검색된 상위 2장을 `image_data_1/2`, `image_mime_1/2` 로 만든다. 결과가 2장보다 적으면 알기 쉬운 에러 메시지를 낸다.
- 비전 체인의 system 프롬프트는 "이미지의 특정 부분을 직접 언급하고, 대화하듯 답하고, 목록은 줄이고, 마크다운으로 강조하라"는 내용이다.

### Part 5. 전체 파이프라인

```python
def ask_fashion_rag(query_ko, n_results=N_RESULTS, show=True):
    query_en     = translate(query_ko, "English")              # ① 번역
    results      = query_db(image_vdb, query_en, n_results)    # ② Retrieval
    prompt_input = format_prompt_inputs(results, query_en)     # ③ Augmentation
    response_en  = vision_chain.invoke(prompt_input)           # ④ Generation
    response_ko  = translate(response_en, "Korean")            # ⑤ 번역
    display(Markdown(response_ko))                             # 마크다운으로 렌더링
    return {...}

ask_fashion_rag("결혼식 하객룩으로 어울리는 원피스 스타일링 알려줘")
```

- 실행 결과: 번역된 검색어는 `"Please tell me about dress styling suitable for wedding guest outfits."` 였고, `image_0096` (0.7471)과 `image_0069` (0.7556)가 검색됐다.
- **대화형 모드:** 노트북에서 `input()` 무한 루프를 바로 돌리면 "전체 실행" 때 멈춘 것처럼 보인다. 그래서 `RUN_INTERACTIVE = False` 를 기본값으로 둔다. 루프 안에서는 `try/except` 로 API 오류가 나도 루프가 멈추지 않게 한다.

---

## 5. 실습 문제 풀이

**실습 1. 검색 개수 바꾸기**

```python
results = query_db(image_vdb, "a black leather jacket", results=4)
print_results(results)
```

**실습 2. 이미지로 이미지 검색 (Image → Image)**

```python
query_img = os.path.join(DATASET_FOLDER, image_files[0])
ex2_results = image_vdb.query(
    query_uris=[query_img],              # 이미지 경로를 '리스트'로 넣는다
    n_results=3,
    include=["uris", "distances"],
)
print_results(ex2_results)
```

- 1위는 **검색에 쓴 그 사진 자신**이고, distance는 **거의 0**이다. 같은 이미지는 같은 벡터가 되기 때문이다.

**실습 3. 페르소나 바꾸기 (가성비 스타일리스트)**

```python
budget_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a budget-friendly fashion stylist. Using the given images, explain how to "
               "recreate a similar look with affordable items. Refer directly to parts of the images, "
               "keep a conversational tone, and use markdown for emphasis."),
    ("user", [ ...비전 체인과 같은 콘텐츠 블록... ]),
])
budget_chain = budget_prompt | ChatOpenAI(model=VISION_MODEL, temperature=0.0) | StrOutputParser()
```

- system 프롬프트만 바꿔도 챗봇의 성격이 달라진다.

**실습 4. 거리 임계값 필터**

```python
def filter_by_distance(results, max_distance):
    kept = []
    for rid, uri, dist in zip(results["ids"][0], results["uris"][0], results["distances"][0]):
        if dist <= max_distance:          # 거리가 작을수록 비슷하므로 '이하'만 남긴다
            kept.append((rid, uri, dist))
    return kept
```

- 관련 없는 사진을 LLM에 보내면 엉뚱한 답이 나온다. 그래서 거리 기준으로 걸러 낸다.

---

## 6. 확인 퀴즈 답

1. **CLIP으로 텍스트로 이미지를 검색할 수 있는 이유:** 대조 학습으로 텍스트와 이미지를 같은 512차원 벡터 공간에 놓았기 때문이다. 텍스트 벡터와 가까운 이미지 벡터를 찾으면 된다.
2. **distance가 작을수록:** 검색어와 더 비슷하다는 뜻이다.
3. **base64로 인코딩하는 이유:** API 요청은 JSON 텍스트라서 바이너리 이미지를 그대로 넣을 수 없다. 문자열로 바꿔 data URL에 담으면 파일을 업로드하지 않고도 보낼 수 있다.
4. **`streaming=True` 의 장점:** 전체(약 3.5GB)를 받지 않고 필요한 만큼만 읽는다. 디스크와 시간을 아낀다.
5. **`__file__` 에서 에러가 나는 이유:** `__file__` 은 `.py` 파일을 실행할 때만 있는 변수다. 노트북에는 없어서 `NameError` 가 난다. 대신 `os.getcwd()` 를 쓴다.
6. **번역을 두 번 하는 이유:** CLIP과 프롬프트가 영어에서 가장 잘 동작해서 질문을 영어로 바꾸고, 사용자에게는 한국어로 보여 주려고 답변을 다시 번역한다.
7. **LCEL 각 단계의 출력 타입:** `ChatPromptTemplate` 은 메시지 목록(`ChatPromptValue`), `ChatOpenAI` 는 `AIMessage`, `StrOutputParser` 는 `str` 을 출력한다.

---

## 7. 내가 알게 된 점 (수업 메모)

- Hugging Face에는 다양한 분야의 데이터셋이 있고, 스트리밍으로 필요한 만큼만 불러올 수 있다.
- `os.path.join(BASE_DIR, "fashion_dataset")` 처럼 경로를 조합해서 새 폴더 위치를 만든다.
- 이미지 여러 장을 한 줄로 보여 줄 때는 캔버스를 만들고 `paste` 로 붙인다.
- 이미지가 이미 있으면 다시 받지 않고, DB에 이미 있으면 다시 넣지 않는다.
- CLIP은 이미지와 텍스트를 같은 공간에 놓지만, 한국어 검색에는 약하다.
- 비슷할수록 코사인 유사도는 1에, Chroma distance는 0에 가까워진다.

---

## 8. 개선 아이디어

- **번역 결과를 검색어 형태로 다듬기:** 실습에서 "Please tell me about dress styling..." 처럼 문장 전체가 검색어로 들어갔다. 번역 프롬프트에 "핵심 키워드만 짧게 뽑아라"를 추가하면 검색 정확도가 올라갈 수 있다.
- **다국어 CLIP** (예: `xlm-roberta-base-ViT-B-32`)을 쓰면 번역 없이 한국어로 바로 검색할 수 있다.
- **메타데이터 필터:** 의류 카테고리를 `metadatas` 로 저장하고 `where={"category": "dress"}` 로 걸러서 검색한다.
- **대화 기억:** `RunnableWithMessageHistory` 로 이전 질문의 맥락을 유지한다.
- **웹 UI:** Gradio나 Streamlit으로 `ask_fashion_rag()` 를 감싸 웹 데모를 만든다.

---

## 한 줄 요약

| 배운 것 | 핵심 |
|---|---|
| 멀티모달 RAG | 이미지를 검색해서 비전 LLM이 답할 근거로 쓴다 |
| CLIP | 텍스트와 이미지를 같은 벡터 공간에 놓아, 텍스트로 이미지를 찾는다 |
| Chroma | `embedding_function` 과 `data_loader` 를 연결하면 경로만 넣어도 임베딩과 저장이 자동으로 된다 |
| LCEL | `prompt \| model \| parser` 로 체인을 만들고 `invoke(dict)` 로 실행한다 |
| 비전 입력 | `{"type": "image_url", "image_url": {"url": "data:<MIME>;base64,..."}}` |