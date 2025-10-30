# app.py — RAG + PASS/FAIL 학습 + Vision 추출
# 패치 포함: 강제 JSON, 키 정규화/일관화, 빈칸 토큰 처리, PASS/FAIL 메타/중복방지,
# 업로드-즉시분석(UX), 스캔PDF 경고, 통합판정 프롬프트 우선순위

import streamlit as st
import os, io, json, base64, glob, re, hashlib, unicodedata, string
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ==============================
# 설정
# ==============================
APP_TITLE = "📄 AI 결재 사전검토 (RAG + PASS/FAIL 학습 통합형)"
DB_DIR = "./chroma_db"
DATASET_PATH = "./pass_fail_dataset.json"
GUIDE_COLLECTION_NAME = "company_guideline"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ==============================
# 초기화
# ==============================
chroma_client = chromadb.PersistentClient(path=DB_DIR)

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

def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = text.replace("\r", "\n")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0: start = 0
        if start >= L: break
    return chunks

def embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    if not texts:
        return []
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    return embedder.embed_documents(texts)

def pil_to_b64(img: Image.Image) -> str:
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
}

def normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
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
    if not chunks or not embeddings:
        st.error(f"{source} PDF에서 텍스트를 찾을 수 없습니다.")
        return
    col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    col.delete(where={"source": source})
    base = 10_000 if source == "caution" else 0
    ids = [f"{source}_{base + i}" for i in range(len(chunks))]
    metas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    st.success(f"{source} {len(chunks)}개 저장 완료 ✅")

def search_guideline(query: str, api_key: str, k: int = 4) -> List[str]:
    col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    q_emb = embedder.embed_query(query)
    result = col.query(query_embeddings=[q_emb], n_results=k)
    if not result or not result.get("documents"):
        return []
    texts = []
    for i in range(len(result["documents"][0])):
        texts.append(result["documents"][0][i])
    return texts

# ==============================
# Vision: 결재 문서 분석 (강제 JSON)
# ==============================
def gpt_extract_table(api_key: str, img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    img_b64 = pil_to_b64(img)

    system_msg = (
        "너는 회사 결재/경비 문서를 표 형태로 분석하는 AI다. "
        "결재선(결재, 승인, 참조 등)은 무시하고, 설명 없이 JSON만 반환해라."
    )
    user_msg = (
        "아래 결재서 이미지를 분석하여 JSON으로 만들어라.\n"
        "필수 키: ['제목','증빙유형','지급요청일','attachment_count']\n"
        "가능하면 숫자는 숫자형으로. 찾지 못한 키는 빈 문자열 또는 0으로 둔다."
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

    # attachment_count 숫자화
    if isinstance(data.get("attachment_count"), str):
        try:
            num = re.findall(r"\d+", data["attachment_count"])
            data["attachment_count"] = int(num[0]) if num else 0
        except Exception:
            data["attachment_count"] = 0

    # 기본값 보정
    data.setdefault("제목", "")
    data.setdefault("증빙유형", "")
    data.setdefault("지급요청일", "")
    data.setdefault("attachment_count", 0)

    return data

# ==============================
# PASS / FAIL 데이터 학습 저장(메타/중복 방지)
# ==============================
def save_pass_fail_data(new_data: List[Dict[str, Any]]):
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

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
def integrated_compare(api_key: str, test_json: Dict[str, Any], model: str = "gpt-4o") -> str:
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, max_tokens=2500)

    rag_texts: List[str] = []
    for q in [
        "지급요청일 입력 규칙",
        "증빙유형 입력 규칙",
        "첨부파일 규칙",
        "경비청구 주의사항"
    ]:
        rag_texts.extend(search_guideline(q, api_key, k=2))

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        pass_data = [d for d in dataset if d.get("상태") == "PASS"]
        fail_data = [d for d in dataset if d.get("상태") == "FAIL"]
    else:
        pass_data, fail_data = [], []

    # 빈칸 우선 체크 로컬 규칙 (오탐 방지)
    local_fail_reasons = []
    # 예시 필수 키(필요 시 추가/수정)
    required_keys = ["제목", "증빙유형", "지급요청일"]
    for k in required_keys:
        if is_empty(test_json.get(k)):
            local_fail_reasons.append(f"필수 항목 '{k}'이(가) 비어 있음")
    # 첨부 규칙(회사 규정에 따라 조정)
    # 예: 지급요청/증빙유형이 있는 문서에서 첨부 0건이면 FAIL
    if not is_empty(test_json.get("증빙유형")) and test_json.get("attachment_count", 0) == 0:
        local_fail_reasons.append("증빙유형이 있는데 attachment_count=0 (첨부 누락 가능)")

    prompt = f"""
너는 회사 결재 문서를 사전 검토하는 AI다.

판정 우선순위:
A. 아래 '로컬 규칙'에서 FAIL 사유가 있으면 반드시 FAIL로 간주하고 사유를 맨 위에 명시한다.
B. 그 외는 [회사 가이드라인/유의사항]과 [PASS/FAIL 예시]를 참고하여 보수적으로 판단한다.

[로컬 규칙에서 감지된 FAIL 사유]
{json.dumps(local_fail_reasons, ensure_ascii=False, indent=2)}

[회사 가이드라인 및 유의사항 일부]
{json.dumps(rag_texts[:10], ensure_ascii=False, indent=2)}

[PASS 문서 예시(일부)]
{json.dumps(pass_data[:5], ensure_ascii=False, indent=2)}

[FAIL 문서 예시(일부)]
{json.dumps(fail_data[:5], ensure_ascii=False, indent=2)}

[검토할 결재 문서(JSON)]
{json.dumps(test_json, ensure_ascii=False, indent=2)}

요구사항:
1) 발견된 문제를 아래 포맷으로 나열한다.
- 항목명: ...
- 문제점: ...
- 수정 예시: ...

2) 맨 마지막 줄에 '최종 판정: PASS' 또는 '최종 판정: FAIL (이유 ...)' 형태로 출력한다.
3) 불필요한 서론은 쓰지 말고 형식만 지켜라.
"""
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)

# ==============================
# UI
# ==============================
with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
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
                text = pdf_to_text(pdf_file.read())
                if text.strip():
                    chunks = split_text(text)
                    embs = embed_texts(chunks, api_key)
                    save_pdf_to_chroma(chunks, embs, "guideline")
            if caution_pdf:
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
            for img_file in pass_imgs or []:
                img = Image.open(img_file).convert("RGB")
                data = gpt_extract_table(api_key, img, model)
                data["상태"] = "PASS"
                learned.append(data)
            for img_file in fail_imgs or []:
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
    if test_img and api_key:
        img = Image.open(test_img).convert("RGB")
        st.image(img, caption="테스트 문서", use_container_width=True)
        with st.spinner("문서 인식 중..."):
            test_json = gpt_extract_table(api_key, img, model)
        # 키 통일
        test_json = normalize_keys(test_json)
        st.session_state["test_json"] = test_json
        st.success("문서 인식 완료 ✅")
        st.code(json.dumps(test_json, ensure_ascii=False, indent=2), language="json")

# ---------------- 오른쪽: 결과 ----------------
with col2:
    st.header("④ AI 통합 검토 결과")
    if st.button("AI 자동 검토 실행"):
        if not api_key:
            st.error("API Key가 필요합니다.")
        else:
            test_json = st.session_state.get("test_json")
            if not test_json:
                st.error("테스트 문서를 먼저 업로드하세요.")
            else:
                with st.spinner("AI가 규정 + 학습 데이터를 종합 분석 중..."):
                    result = integrated_compare(api_key, test_json, model)
                st.success("검토 완료 ✅")
                st.markdown("**검토 결과 (AI 분석)**")
                st.write(result)
