# app.py — RAG + PASS/FAIL 예시 기반 + 유사도 top-k 선별
# 포함: 강제 JSON, 키 정규화/일관화, 빈칸 토큰, PASS/FAIL 메타·중복 방지,
# 업로드-즉시 분석, 스캔PDF 경고, 로컬 룰 우선, 임베딩 유사도 기반 예시 선별

import streamlit as st
import os, io, json, base64, glob, re, hashlib, unicodedata, string
from datetime import datetime
from typing import List, Dict, Any, Tuple
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import numpy as np

# ==============================
# 설정
# ==============================
APP_TITLE = "📄 AI 결재 사전검토"
DB_DIR = "./chroma_db"
DATASET_PATH = "./pass_fail_dataset.json"
GUIDE_COLLECTION_NAME = "company_guideline"
TOPK_SIMILAR = 3  # PASS/FAIL 각각 top-k

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
    if not texts: return []
    emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    return emb.embed_documents(texts)

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
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
}

def normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[ALIASES.get(k, k)] = v
    return out

EMPTY_TOKENS = {"", "-", "—", "–", "ㅡ", "없음", "무", "n/a", "na", "null", "none", "미입력", "미기재", "해당없음"}

def is_empty(v) -> bool:
    """숫자 0은 값으로 간주(예: 첨부 0건), 그 외 토큰/공백/구두점만이면 빈칸"""
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return False
    s = str(v).strip().lower()
    if s in EMPTY_TOKENS:
        return True
    if all((ch in string.punctuation) or ch.isspace() for ch in s):
        return True
    return False

def dict_to_sorted_text(d: Dict[str, Any]) -> str:
    """임베딩용 문자열 변환: 키 정렬 후 'k: v' 줄로 합침"""
    try:
        items = sorted(d.items(), key=lambda x: str(x[0]))
        return "\n".join(f"{k}: {v}" for k, v in items if not str(k).startswith("_"))
    except Exception:
        return json.dumps(d, ensure_ascii=False, sort_keys=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

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
    emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    q_emb = emb.embed_query(query)
    result = col.query(query_embeddings=[q_emb], n_results=k)
    if not result or not result.get("documents"): return []
    return [result["documents"][0][i] for i in range(len(result["documents"][0]))]

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
        response_format={"type": "json_object"},  # 강제 JSON
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url", "image_url": {"url": img_b64}},
            ]},
        ],
    )

    data = json.loads(resp.choices[0].message.content)
    data = normalize_keys(data)

    # attachment_count 숫자화
    if isinstance(data.get("attachment_count"), str):
        try:
            nums = re.findall(r"\d+", data["attachment_count"])
            data["attachment_count"] = int(nums[0]) if nums else 0
        except Exception:
            data["attachment_count"] = 0

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

    seen = {d.get("_hash") for d in existing if d.get("_hash")}
    saved = 0
    for d in new_data:
        d = normalize_keys(d)
        meta_str = json.dumps(d, ensure_ascii=False, sort_keys=True)
        _hash = hashlib.md5(meta_str.encode()).hexdigest()
        if _hash in seen:  # 중복 방지
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
# 유사 샘플 선별 (임베딩 top-k)
# ==============================
def select_similar_examples(api_key: str, test_json: Dict[str, Any], dataset: List[Dict[str, Any]], k: int = TOPK_SIMILAR) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not dataset: return [], []
    emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

    # 테스트 문서 임베딩
    test_text = dict_to_sorted_text(test_json)
    try:
        test_vec = np.array(emb.embed_query(test_text), dtype=float)
    except Exception:
        return [], []

    # PASS/FAIL 분리 후 각각 유사도 top-k
    pass_docs = [d for d in dataset if d.get("상태") == "PASS"]
    fail_docs = [d for d in dataset if d.get("상태") == "FAIL"]

    def topk_for(group):
        scored = []
        for d in group:
            vec = np.array(emb.embed_query(dict_to_sorted_text(d)), dtype=float)
            sim = cosine_sim(test_vec, vec)
            scored.append((sim, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]

    return topk_for(pass_docs), topk_for(fail_docs)

# ==============================
# 통합 비교 (가이드라인 + 패턴 학습 + 유사도 예시)
# ==============================
def integrated_compare(api_key: str, test_json: Dict[str, Any], model: str = "gpt-4o") -> str:
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, max_tokens=2500)

    # RAG 스니펫 수집
    rag_texts: List[str] = []
    for q in ["지급요청일 입력 규칙", "증빙유형 입력 규칙", "첨부파일 규칙", "경비청구 주의사항"]:
        rag_texts.extend(search_guideline(q, api_key, k=2))

    # 데이터셋 로드 및 유사 예시 선별
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = []

    # 로컬 룰: 필수 항목/첨부 누락 전부 기록
    local_fail_reasons = []
    required_keys = ["제목", "증빙유형", "지급요청일", "attachment_count"]
    for k in required_keys:
        if is_empty(test_json.get(k)):
            local_fail_reasons.append(f"필수 항목 '{k}'이(가) 비어 있음")
    if not is_empty(test_json.get("증빙유형")) and test_json.get("attachment_count", 0) == 0:
        local_fail_reasons.append("증빙유형 존재하지만 첨부 0건 (첨부 누락 가능)")

    # 유사 PASS/FAIL top-k 선별
    pass_topk, fail_topk = select_similar_examples(api_key, test_json, dataset, k=TOPK_SIMILAR)

    prompt = f"""
너는 회사 결재 문서를 사전 검토하는 AI다.

반드시 아래 조건을 지켜라:
- 문서의 모든 항목을 끝까지 검토하라. 하나의 FAIL 사유를 발견해도 멈추지 말고, **모든 문제를 전부 나열**해야 한다.
- 각 문제는 개별 항목별로 구체적인 이유와 수정 예시를 포함하라.

판정 우선순위:
A. 아래 '로컬 규칙'에서 FAIL 사유가 있으면 반드시 FAIL로 간주하고, 모든 항목에 대한 문제점을 나열한다.
B. 그 외 항목도 PASS/FAIL 예시 및 가이드라인을 참고하여 추가적인 누락·오류가 있으면 함께 제시한다.

[로컬 규칙에서 감지된 FAIL 사유]
{json.dumps(local_fail_reasons, ensure_ascii=False, indent=2)}

[회사 가이드라인 및 유의사항 일부]
{json.dumps(rag_texts[:10], ensure_ascii=False, indent=2)}

[유사 PASS 문서 예시(최대 {TOPK_SIMILAR}개)]
{json.dumps(pass_topk, ensure_ascii=False, indent=2)}

[유사 FAIL 문서 예시(최대 {TOPK_SIMILAR}개)]
{json.dumps(fail_topk, ensure_ascii=False, indent=2)}

[검토할 결재 문서(JSON)]
{json.dumps(test_json, ensure_ascii=False, indent=2)}

요구사항:
1) 모든 항목을 끝까지 검토하여 문제를 빠짐없이 나열한다.
2) 출력은 아래 형식으로만 써라.
- 항목명: ...
- 문제점: ...
- 수정 예시: ...
3) 맨 마지막 줄에 '최종 판정: PASS' 또는 '최종 판정: FAIL (이유 ...)' 형태로 작성한다.
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
    st.caption("가이드라인 PDF 임베딩 → PASS/FAIL 학습(선택) → 테스트 문서 업로드 순서로 진행하세요.")

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

    st.header("① 결재 문서 업로드")
    test_img = st.file_uploader("검토할 문서 (jpg/png)", type=["jpg", "png", "jpeg"])

    if test_img and api_key:
    # 이미지 열기
        img = Image.open(test_img).convert("RGB")
        st.image(img, caption="업로드 문서", use_container_width=True)

    # Vision 호출로 JSON 추출
        with st.spinner("문서 인식 중..."):
            doc_json = gpt_extract_table(api_key, img, model="gpt-4o")  # ← 여기서 doc_json 생성

        st.success("문서 인식 완료 ✅")

    # UI에 JSON은 보여주지 않고 세션에만 저장
        st.session_state["doc_json"] = doc_json

# ---------------- 오른쪽: 결과 ----------------
with col2:
    st.header("④ AI 통합 검토 결과")
    if st.button("AI 검토 실행"):
    if not api_key:
        st.error("OpenAI API Key를 입력하세요.")
    else:
        doc_json = st.session_state.get("doc_json")   # 세션에서 안전하게 꺼냄
        if not doc_json:
            st.error("먼저 문서를 업로드해 주세요.")
        else:
            with st.spinner("AI가 문서를 검토 중..."):
                result = integrated_compare(api_key, doc_json, model="gpt-4o")
            st.session_state["analysis_result"] = result
            st.success("검토 완료 ✅")
            st.write(result)
    
