# app.py — RAG + PASS/FAIL 학습 + Vision 추출
# 패치 포함: 강제 JSON, 키 정규화/일관화, 빈칸 토큰 처리, PASS/FAIL 메타/중복방지,
# 업로드-즉시분석(UX), 스캔PDF 경고, 통합판정 프롬프트 우선순위, 로컬 FAIL 규칙 강화 (날짜, 금액, 필수 첨부)

import streamlit as st
import os, io, json, base64, glob, re, hashlib, unicodedata, string
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==============================
# 설정
# ==============================
APP_TITLE = "📄 AI 결재 사전검토 (RAG + PASS/FAIL 학습 통합형)"
# 🚨 클라우드 환경(Streamlit Cloud)에서 파일 시스템 권한 오류(Read-only)를 방지하기 위해
# 🚨 /tmp 디렉토리를 사용합니다. 이 디렉토리는 쓰기가 가능하지만, 앱 재시작 시 데이터는 초기화됩니다.
DB_DIR = "/tmp/chroma_db"
DATASET_PATH = "/tmp/pass_fail_dataset.json" # 데이터셋 파일 경로도 /tmp로 변경
GUIDE_COLLECTION_NAME = "company_guideline"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ==============================
# 초기화
# ==============================
# ChromaDB 클라이언트 초기화
try:
    # /tmp 경로에 데이터베이스를 생성 시도
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
except Exception as e:
    # 초기화 실패 시 (예: 클라우드 환경의 권한 문제), 클라이언트 비활성화
    st.error(f"ChromaDB 초기화 오류 (파일 시스템 권한 문제 예상): {e}")
    st.warning(f"ChromaDB 경로를 '{DB_DIR}'로 변경했습니다. 만약 오류가 지속된다면 임베딩/RAG 기능은 동작하지 않습니다.")
    chroma_client = None
    
# 세션 상태 초기화
if "test_json" not in st.session_state:
    st.session_state["test_json"] = None
if "local_fail_reasons" not in st.session_state:
    st.session_state["local_fail_reasons"] = []


# ==============================
# 공통 유틸
# ==============================
def pdf_to_text(file: bytes) -> str:
    """텍스트 기반 PDF는 바로 추출, 스캔 PDF면 빈 문자열 가능"""
    try:
        reader = PdfReader(io.BytesIO(file))
        texts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(texts)
        if not text.strip():
            st.warning("⚠️ PDF에서 텍스트가 추출되지 않았습니다. 스캔 PDF일 수 있어요.")
        return text
    except Exception as e:
        st.error(f"PDF 읽기 오류: {e}")
        return ""

def split_text(text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
    """텍스트를 적절한 크기로 분할하여 임베딩 청크로 만듬 (chunk_size를 1500으로 상향)"""
    text = text.replace("\r", "\n")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        
        # 중첩(overlap)을 적용하여 다음 시작 지점을 계산
        start = end - overlap
        if start < 0: start = 0
        if start >= L: break
    return chunks

def embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    """OpenAI 임베딩 모델을 사용하여 텍스트 임베딩 생성 (모델: text-embedding-3-small)"""
    if not texts:
        return []
    try:
        # 모델을 text-embedding-3-small로 변경하여 속도 향상
        embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        return embedder.embed_documents(texts)
    except Exception as e:
        st.error(f"임베딩 생성 오류: {e}")
        return []

def pil_to_b64(img: Image.Image) -> str:
    """PIL Image 객체를 base64 인코딩된 문자열로 변환 (GPT Vision API용)"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# ==============================
# 키 정규화 / 빈칸 판정
# ==============================
ALIASES = {
    # 첨부/증빙 관련 → attachment_count로 통일
    "첨부파일수": "attachment_count",
    "첨부": "attachment_count",
    "증빙개수": "attachment_count",
    "첨부(건수)": "attachment_count",
    # 기타 자주 보이는 한글 키 치환을 여기에 계속 보강해도 됨
    "청구금액": "금액",
    "총금액": "금액",
}

def normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 키를 표준화된 키로 변환"""
    out = {}
    for k, v in d.items():
        nk = ALIASES.get(k, k)
        out[nk] = v
    return out

EMPTY_TOKENS = {"", "-", "—", "–", "ㅡ", "없음", "무", "n/a", "na", "null", "none", "미입력", "미기재", "해당없음"}

def is_empty(v) -> bool:
    """숫자 0은 값이 있는 것으로 간주(예: 첨부 0건), 그 외 토큰/공백은 빈칸"""
    if v is None:
        return True
    if isinstance(v, (int, float)):
        # 0.0, 0, -0.0 등은 값이 있는 것으로 간주. 
        # 금액 0원 결재가 허용되지 않는다면 로컬 규칙에서 따로 처리해야 함.
        return False 
    s = str(v).strip().lower()
    if s in EMPTY_TOKENS:
        return True
    # 구두점/공백만으로 이루어졌다면 빈 칸
    if all((ch in string.punctuation) or ch.isspace() for ch in s):
        return True
    return False

# ==============================
# PDF → Chroma 저장 / 검색
# ==============================
def save_pdf_to_chroma(chunks: List[str], embeddings: List[List[float]], source: str):
    """PDF 텍스트 청크를 ChromaDB에 저장"""
    if not chroma_client:
        st.error("ChromaDB가 초기화되지 않았습니다.")
        return

    if not chunks or not embeddings:
        st.error(f"{source} PDF에서 텍스트를 찾을 수 없습니다.")
        return
    
    col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    
    # 이전 데이터 삭제 후 저장 (중복 방지)
    col.delete(where={"source": source}) 
    
    base = 10_000 if source == "caution" else 0
    ids = [f"{source}_{base + i}" for i in range(len(chunks))]
    metas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    
    col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    st.success(f"{source} {len(chunks)}개 저장 완료 ✅")

def search_guideline(query: str, api_key: str, k: int = 4) -> List[str]:
    """ChromaDB에서 가이드라인 관련 텍스트 검색"""
    if not chroma_client:
        return ["ChromaDB가 초기화되지 않아 가이드라인을 검색할 수 없습니다."]

    try:
        col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
        # 검색 시에도 동일하게 text-embedding-3-small 사용
        embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key) 
        q_emb = embedder.embed_query(query)
        result = col.query(query_embeddings=[q_emb], n_results=k)
        
        if not result or not result.get("documents"):
            return []
        
        texts = []
        for doc_list in result["documents"]:
            texts.extend(doc_list)
        return texts
    except Exception as e:
        # st.error(f"가이드라인 검색 오류: {e}") # 디버깅용
        return [f"가이드라인 검색 중 오류 발생: {str(e)}"]

# ==============================
# Vision: 결재 문서 분석 (강제 JSON)
# ==============================
def gpt_extract_table(api_key: str, img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    """GPT Vision을 사용하여 이미지에서 핵심 결재 정보를 JSON으로 추출"""
    client = OpenAI(api_key=api_key)
    img_b64 = pil_to_b64(img)

    system_msg = (
        "너는 회사 결재/경비 문서를 표 형태로 분석하는 AI다. "
        "결재선(결재, 승인, 참조 등)은 무시하고, 설명 없이 JSON만 반환해라."
    )
    user_msg = (
        "아래 결재서 이미지를 분석하여 JSON으로 만들어라.\n"
        "필수 키: ['제목','증빙유형','지급요청일','금액','attachment_count']\n" # ⬅ 금액 추가
        "금액은 원화 기호 없이 숫자형(int/float)으로 추출한다. 찾지 못한 키는 빈 문자열 또는 0으로 둔다. attachment_count는 첨부 건수를 숫자형으로 추출한다."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},  # ⬅ 강제 JSON
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url", "image_url": {"url": img_b64}},
            ]},
        ],
    )

    data = json.loads(resp.choices[0].message.content)

    # 키 통일/정규화
    data = normalize_keys(data)

    # attachment_count 숫자화 및 기본값 보정
    for key in ["attachment_count", "금액"]:
        if isinstance(data.get(key), str):
            try:
                # 숫자만 추출 (ex: '3건' -> 3, '1,000,000원' -> 1000000)
                num_str = re.sub(r'[^0-9\.\-]', '', data[key].replace(',', ''))
                if num_str:
                    if '.' in num_str:
                        data[key] = float(num_str)
                    else:
                        data[key] = int(num_str)
                else:
                    data[key] = 0 if key == "attachment_count" else ""
            except Exception:
                data[key] = 0 if key == "attachment_count" else ""

    # 기본값 보정
    data.setdefault("제목", "")
    data.setdefault("증빙유형", "")
    data.setdefault("지급요청일", "")
    data.setdefault("금액", "") # 금액은 문자열로 남겨서 is_empty 체크에 포함시킬 수도 있음
    data.setdefault("attachment_count", 0)

    return data

# ==============================
# 로컬 규칙 검사 (FAIL 명확화 핵심)
# ==============================
def check_local_rules(test_json: Dict[str, Any]) -> List[str]:
    """LLM 호출 전, 코드 기반의 명시적인 FAIL 규칙을 검사"""
    local_fail_reasons = []

    # 1. 필수 키 빈칸 검사 (FAIL 확정 1순위)
    # 회사 규정에 따라 금액, 거래처 등 추가
    required_keys = ["제목", "증빙유형", "지급요청일", "금액"] 
    for k in required_keys:
        if is_empty(test_json.get(k)):
            local_fail_reasons.append(f"필수 항목 '{k}'이(가) 비어 있음")

    # 2. 날짜 형식 유효성 검사 (YYYY-MM-DD 형식 예상)
    payment_date_str = str(test_json.get("지급요청일", "")).strip()
    if payment_date_str:
        # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD 등 일반적인 날짜 형식 검사
        if not re.match(r"^\d{4}[\-\/\.]\d{1,2}[\-\/\.]\d{1,2}$", payment_date_str):
             local_fail_reasons.append(f"'지급요청일'의 형식 오류: '{payment_date_str}' (YYYY-MM-DD 등 규정된 형식 불일치)")
        else:
            # 유효한 날짜인지 추가 검사 (예: 2023-13-40 같은 오류 방지)
            try:
                # 첫 번째 매칭된 날짜 형식으로 파싱 시도 (분리 기호 통일)
                date_parts = re.split(r'[\-\/\.]', payment_date_str)
                if len(date_parts) == 3:
                     datetime(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
            except ValueError:
                local_fail_reasons.append(f"'지급요청일'의 날짜 값 오류: '{payment_date_str}' (유효하지 않은 날짜 값)")


    # 3. 금액 유효성 검사 (음수/비정상 값)
    amount = test_json.get("금액")
    if isinstance(amount, (int, float)):
        if amount < 0:
            local_fail_reasons.append(f"'금액'이 음수임: {amount} (비정상 값)")
    # 금액이 있어야 하는 문서인데 금액이 0원이라면 경고/FAIL 처리 (회사 규정 따라 조정)
    elif amount == 0 and not is_empty(test_json.get("증빙유형")):
        local_fail_reasons.append(f"'금액'이 0원임: {amount} (금액 누락 또는 0원 결재 불가)")

    # 4. 필수 첨부 규칙 (회사 규정에 따라 조정)
    # 예: 지급요청/증빙유형이 있는 문서 중 특정 유형은 첨부 0건이면 FAIL
    must_attach_types = ["카드영수증", "세금계산서", "세금 계산서", "송금확인증", "계약서"] # ⬅ 첨부 필수 증빙 유형
    doc_type = test_json.get("증빙유형", "").strip()

    if doc_type in must_attach_types and test_json.get("attachment_count", 0) == 0:
        local_fail_reasons.append(f"증빙유형 '{doc_type}'은(는) 첨부 필수 항목인데 attachment_count=0 (첨부 누락)")

    return local_fail_reasons

# ==============================
# PASS / FAIL 데이터 학습 저장(메타/중복 방지)
# ==============================
def save_pass_fail_data(new_data: List[Dict[str, Any]]):
    """학습 데이터를 JSON 파일에 저장 (중복 방지 및 메타데이터 추가)"""
    # /tmp 경로에 파일이 존재하지 않으면 빈 파일 생성
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
             existing = []
             st.warning(f"기존 {DATASET_PATH} 파일이 손상되어 초기화합니다.")

    # 중복 방지용 해시 집합
    seen = {d.get("_hash") for d in existing if d.get("_hash")}

    saved = 0
    for d in new_data:
        d = normalize_keys(d)
        meta_str = json.dumps(d, ensure_ascii=False, sort_keys=True)
        _hash = hashlib.md5(meta_str.encode()).hexdigest()
        if _hash in seen:
            continue
        d["_hash"] = _hash
        d["_saved_at"] = datetime.now().isoformat(timespec="seconds")
        d["_doc_type"] = (d.get("제목") or "").split()[0]
        existing.append(d)
        seen.add(_hash)
        saved += 1

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    st.success(f"PASS/FAIL 데이터 {saved}건 저장 완료 (총 {len(existing)}건) ✅")

# ==============================
# 통합 비교 (가이드라인 + 패턴 학습)
# ==============================
def integrated_compare(api_key: str, test_json: Dict[str, Any], local_fail_reasons: List[str], model: str = "gpt-4o") -> str:
    """RAG와 PASS/FAIL 학습 데이터를 종합하여 최종 판정"""
    llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key, max_tokens=2500) # temperature를 0.1로 약간 높여 추론 유연성 확보

    # RAG 검색 (가이드라인)
    rag_texts: List[str] = []
    # 다양한 쿼리를 통해 가이드라인 청크를 검색
    for q in [
        f"'{test_json.get('제목', '문서')}'에 대한 규정",
        f"'{test_json.get('증빙유형', '증빙')}' 처리 규칙",
        "지급요청일 형식 및 규칙",
        "첨부파일 규칙",
        "경비청구 주의사항"
    ]:
        rag_texts.extend(search_guideline(q, api_key, k=2))
    
    # 중복 제거
    rag_texts = list(set(rag_texts))

    # PASS/FAIL 학습 데이터 로드
    # 파일이 존재하지 않으면 빈 파일 생성 (Streamlit Cloud에서 첫 실행 시 파일이 없기 때문)
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            try:
                dataset = json.load(f)
            except json.JSONDecodeError:
                dataset = []

        pass_data = [d for d in dataset if d.get("상태") == "PASS"]
        fail_data = [d for d in dataset if d.get("상태") == "FAIL"]
    else:
        pass_data, fail_data = [], []


    # LLM 프롬프트 구성 (FAIL 우선순위 명확화)
    prompt = f"""
너는 회사 결재 문서를 사전 검토하는 AI다.

[가장 중요한 원칙: 로컬 규칙 FAIL은 최종 FAIL]
1. 아래 '로컬 규칙에서 감지된 FAIL 사유'에 항목이 **단 하나라도** 있으면, 다른 어떤 정보(가이드라인, PASS/FAIL 예시 등)보다 우선하여 **최종 판정은 반드시 'FAIL'**로 내려야 한다.
2. 로컬 FAIL 사유가 없다면, [회사 가이드라인/유의사항]과 [PASS/FAIL 예시]를 참고하여 보수적으로 판단한다.

[로컬 규칙에서 감지된 FAIL 사유] # ⬅ 명시적 코드 검증 FAIL 사유
{json.dumps(local_fail_reasons, ensure_ascii=False, indent=2)}

[회사 가이드라인 및 유의사항 일부] # ⬅ RAG로 검색된 참고 자료
{json.dumps(rag_texts[:10], ensure_ascii=False, indent=2)}

[PASS 문서 예시(일부)] # ⬅ 학습 패턴
{json.dumps(pass_data[:5], ensure_ascii=False, indent=2)}

[FAIL 문서 예시(일부)] # ⬅ 학습 패턴
{json.dumps(fail_data[:5], ensure_ascii=False, indent=2)}

[검토할 결재 문서(JSON)]
{json.dumps(test_json, ensure_ascii=False, indent=2)}

요구사항:
1) 발견된 문제를 아래 포맷으로 나열한다. 로컬 FAIL 사유가 있다면 그것을 맨 위에 명시한다.
- 항목명: ...
- 문제점: ...
- 수정 예시: ...

2) 맨 마지막 줄에 '최종 판정: PASS' 또는 '최종 판정: FAIL (이유 ...)' 형태로 출력한다. 로컬 FAIL 사유가 있을 경우 해당 사유를 반드시 포함하여 최종 판정을 내린다.
3) 불필요한 서론은 쓰지 말고 형식만 지켜라.
"""
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        return res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        return f"❌ LLM 호출 중 오류 발생: {e}"


# ==============================
# UI
# ==============================
with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
    # API 키를 환경 변수에서 가져오도록 유지
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    model = st.selectbox("모델 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    st.caption("가이드라인 PDF 임베딩 생성 → PASS/FAIL 학습(선택) → 테스트 문서 업로드 순서로 진행하세요.")

col1, col2 = st.columns([1.1, 0.9])

# ---------------- 왼쪽: 데이터 업로드 ----------------
with col1:
    st.header("① 가이드라인 / 유의사항 업로드")
    pdf_file = st.file_uploader("가이드라인 PDF", type=["pdf"], key="guide_pdf")
    caution_pdf = st.file_uploader("유의사항 PDF", type=["pdf"], key="caution_pdf")

    if st.button("PDF 임베딩 생성/업데이트"):
        if not api_key:
            st.error("API Key가 필요합니다.")
        else:
            if pdf_file:
                st.info(f"가이드라인 PDF ({pdf_file.name}) 처리 중...")
                text = pdf_to_text(pdf_file.read())
                if text.strip():
                    chunks = split_text(text)
                    embs = embed_texts(chunks, api_key)
                    save_pdf_to_chroma(chunks, embs, "guideline")
            
            if caution_pdf:
                st.info(f"유의사항 PDF ({caution_pdf.name}) 처리 중...")
                text = pdf_to_text(caution_pdf.read())
                if text.strip():
                    chunks = split_text(text)
                    embs = embed_texts(chunks, api_key)
                    save_pdf_to_chroma(chunks, embs, "caution")

    st.header("② PASS / FAIL 학습 데이터 업로드 (선택)")
    pass_imgs = st.file_uploader("✅ PASS 문서 (여러장 가능)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    fail_imgs = st.file_uploader("❌ FAIL 문서 (여러장 가능)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if st.button("PASS/FAIL 데이터 학습"):
        if not api_key:
            st.error("API Key가 필요합니다.")
        else:
            learned = []
            
            # PASS 문서 처리
            for img_file in pass_imgs or []:
                st.info(f"PASS 문서 인식 중: {img_file.name}")
                img = Image.open(img_file).convert("RGB")
                data = gpt_extract_table(api_key, img, model)
                data["상태"] = "PASS"
                learned.append(data)
                
            # FAIL 문서 처리
            for img_file in fail_imgs or []:
                st.info(f"FAIL 문서 인식 중: {img_file.name}")
                img = Image.open(img_file).convert("RGB")
                data = gpt_extract_table(api_key, img, model)
                data["상태"] = "FAIL"
                learned.append(data)
                
            if learned:
                save_pass_fail_data(learned)
            else:
                st.info("업로드된 PASS/FAIL 이미지가 없습니다.")

    st.header("③ 테스트 결재 문서 업로드")
    test_img = st.file_uploader("테스트 문서", type=["jpg", "png", "jpeg"], key="test_img")
    
    # 테스트 문서 업로드 시 즉시 분석 실행 (UX 개선)
    if test_img and api_key:
        try:
            img = Image.open(test_img).convert("RGB")
            st.image(img, caption="테스트 문서", use_container_width=True)
            
            with st.spinner("문서 인식 중 (GPT-Vision)..."):
                test_json = gpt_extract_table(api_key, img, model)
            
            # 키 통일
            test_json = normalize_keys(test_json)
            st.session_state["test_json"] = test_json
            st.session_state["local_fail_reasons"] = check_local_rules(test_json) # ⬅ 로컬 규칙 즉시 검사
            
            st.success("문서 인식 완료 및 로컬 규칙 검사 완료 ✅")
            st.markdown("**문서 인식 결과 (JSON)**")
            st.code(json.dumps(test_json, ensure_ascii=False, indent=2), language="json")

        except Exception as e:
             st.error(f"문서 인식 중 오류 발생: {e}")
             st.session_state["test_json"] = None
             st.session_state["local_fail_reasons"] = []
    
    elif test_img and not api_key:
        st.error("테스트 문서를 분석하려면 OpenAI API Key를 입력해야 합니다.")
        st.session_state["test_json"] = None
        st.session_state["local_fail_reasons"] = []

# ---------------- 오른쪽: 결과 ----------------
with col2:
    st.header("④ AI 통합 검토 결과")
    
    test_json_data = st.session_state.get("test_json")
    local_fails = st.session_state.get("local_fail_reasons", [])

    if local_fails and test_json_data:
        st.error("🚨 **로컬 규칙 위반 감지:**")
        st.warning("아래의 명시적 FAIL 사유가 있어 최종 판정은 **FAIL**이 될 가능성이 매우 높습니다.")
        st.markdown("\n".join([f"- **{reason}**" for reason in local_fails]))
        st.markdown("---") # 시각적 구분
    elif test_json_data:
        st.info("로컬 규칙 검사: 명시적인 FAIL 사유가 감지되지 않았습니다. 이제 AI 통합 분석을 실행하세요.")
        st.markdown("---")


    if st.button("AI 자동 검토 실행", disabled=not test_json_data):
        if not api_key:
            st.error("API Key가 필요합니다.")
        elif not test_json_data:
            st.error("테스트 문서를 먼저 업로드하고 인식 결과를 확인하세요.")
        else:
            with st.spinner("AI가 규정 + 학습 데이터를 종합 분석 중... (RAG + Pattern Learning)"):
                result = integrated_compare(api_key, test_json_data, local_fails, model)
            
            st.success("검토 완료 ✅")
            st.markdown("#### **검토 결과 (AI 분석)**")
            
            # 최종 판정에 따라 시각적 강조
            if "최종 판정: FAIL" in result:
                st.error(result)
            else:
                st.success(result)
            
            # st.write(result)


# ==============================
# 데이터셋 확인용 (선택적)
# ==============================
if st.checkbox("PASS/FAIL 학습 데이터 목록 보기"):
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            try:
                dataset = json.load(f)
                st.dataframe(dataset)
            except json.JSONDecodeError:
                st.error("PASS/FAIL 데이터 파일 읽기 오류")
    else:
        st.info("저장된 PASS/FAIL 학습 데이터가 없습니다.")
