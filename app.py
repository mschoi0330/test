import streamlit as st
import os
import io
import base64
import json
import hashlib
import re
from typing import List, Dict, Any

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


APP_TITLE = "📄 AI 결재 사전검토"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# -------------------- Chroma 초기화 --------------------
DB_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_DIR)
GUIDE_COLLECTION_NAME = "company_guideline"


# -------------------- PDF → 텍스트 --------------------
def pdf_to_text(file: bytes) -> str:
    reader = PdfReader(io.BytesIO(file))
    texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(texts)


# -------------------- 텍스트 → 청크 --------------------
def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = text.replace("\r", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


# -------------------- 임베딩 --------------------
def embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    if not texts:
        return []
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    return embedder.embed_documents(texts)


# -------------------- Chroma 저장 --------------------
def save_guideline_to_chroma(chunks: List[str], embeddings: List[List[float]]):
    if not chunks or not embeddings:
        st.error("가이드라인 PDF에서 텍스트를 못 뽑았어요.")
        return
    col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    col.delete(where={"source": "guideline"})
    ids = [f"guide_{i}" for i in range(len(chunks))]
    metas = [{"source": "guideline", "chunk": i} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    st.success(f"가이드라인 {len(chunks)}개 저장 완료 ✅")


def save_caution_to_chroma(chunks: List[str], embeddings: List[List[float]]):
    if not chunks or not embeddings:
        st.error("유의사항 PDF에서 텍스트를 못 뽑았어요.")
        return
    col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    col.delete(where={"source": "caution"})
    base = 10_000
    ids = [f"caution_{base + i}" for i in range(len(chunks))]
    metas = [{"source": "caution", "chunk": i} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    st.success(f"유의사항 {len(chunks)}개 저장 완료 ✅")


# -------------------- Chroma 검색 --------------------
def search_guideline(query: str, api_key: str, k: int = 4) -> List[Dict[str, Any]]:
    col = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    q_emb = embedder.embed_query(query)
    result = col.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for i in range(len(result["documents"][0])):
        docs.append({"text": result["documents"][0][i], "metadata": result["metadatas"][0][i]})
    return docs


# -------------------- Vision 유틸 --------------------
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ===== 인식 품질 개선: 전처리 + 스키마 고정 + 2패스 재시도 =====
def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    img = pil.convert("L")  # grayscale
    w, h = img.size
    scale = 1600 / max(w, h)
    if scale > 1.05:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 190 else (0 if p < 120 else p))
    return img.convert("RGB")


FIELD_SCHEMA = {
    "제목": {"required": True, "pattern": r".{2,}"},
    "attachment_count": {"required": True, "pattern": r"^\d+$"},
    "회사": {"required": True, "pattern": r".{2,}"},
    "사용부서(팀)": {"required": True, "pattern": r".{1,}"},
    "사용자": {"required": True, "pattern": r".{1,}"},
    "지급처": {"required": False, "pattern": r".*"},
    "업무추진비": {"required": False, "pattern": r"^\d{1,3}(,\d{3})*$"},
    "결의금액": {"required": False, "pattern": r"^\d{1,3}(,\d{3})*$"},
    "지급요청일": {"required": False, "pattern": r"^\d{4}-\d{2}-\d{2}(\([^)]+\))?$"},
}

KEY_NORMALIZER = {
    "사용부서": "사용부서(팀)",
    "부서": "사용부서(팀)",
    "제 목": "제목",
    "제목 ": "제목",
    "합계": "결의금액",
    "총합계": "결의금액",
}


def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        k2 = KEY_NORMALIZER.get(k.strip(), k.strip())
        out[k2] = v
    return out


def _normalize_values(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)

    def norm_money(s):
        s = str(s).strip()
        s = re.sub(r"[^\d,]", "", s)  # "150,000원" -> "150,000"
        if s.isdigit():
            s = f"{int(s):,}"
        return s

    if "업무추진비" in out:
        out["업무추진비"] = norm_money(out["업무추진비"])
    if "결의금액" in out:
        out["결의금액"] = norm_money(out["결의금액"])

    if "지급요청일" in out:
        s = str(out["지급요청일"])
        s2 = re.sub(r"[./]", "-", s)  # 2025.11.06 → 2025-11-06
        m = re.search(r"(\d{4}-\d{2}-\d{2})(\([^)]+\))?", s2)
        if m:
            out["지급요청일"] = m.group(0)

    if "attachment_count" in out:
        m = re.search(r"\d+", str(out["attachment_count"]))
        out["attachment_count"] = int(m.group()) if m else 0

    return out


def _validate_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    notes = []
    for k, rule in FIELD_SCHEMA.items():
        if rule["required"] and (k not in d or str(d[k]).strip() == ""):
            notes.append(f"필수값 누락: {k}")
        if k in d and rule.get("pattern"):
            if not re.fullmatch(rule["pattern"], str(d[k])):
                notes.append(f"형식 불일치: {k}={d[k]}")
    d["_notes"] = notes
    return d


def _ask_vision(api_key: str, pil_img: Image.Image, model: str, strict_json=True) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    system_msg = (
        "너는 회사 결재/경비 문서 이미지를 읽어 정해진 JSON 스키마로만 반환하는 AI다. 설명 없이 JSON만 출력한다."
    )
    user_msg = (
        "다음 키만 포함하는 JSON을 반환해. 키는 정확히 아래와 일치해야 한다.\n"
        + list(FIELD_SCHEMA.keys()).__str__()
        + "\n규칙:\n"
        "1) 표에 '제목' 셀이 있으면 그 값을 '제목'에. 없으면 상단 큰 제목을 사용.\n"
        "2) 결재/합의/승인/참조/수신 영역은 무시. 필요 시 'approval_line_ignored': true를 추가 가능.\n"
        "3) 'attachment_count'는 숫자만. 해당 칸이 없으면 0.\n"
        "4) '업무추진비','결의금액'은 숫자와 콤마만(예: 150,000).\n"
        "5) '지급요청일'은 YYYY-MM-DD 또는 YYYY-MM-DD(요일) 형식.\n"
        "6) JSON만 출력해."
    )

    kwargs = dict(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": b64}},
                ],
            },
        ],
    )
    if strict_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return {"_raw": content}


def gpt_extract_table(api_key: str, pil_img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    개선된 2-패스 추출:
    - 이미지 전처리
    - 강제 JSON + 스키마 고정
    - mini ↔ 4o 교차 재시도 후 사후 보정/검증
    """
    img = preprocess_for_ocr(pil_img)

    # 1차
    data1 = _ask_vision(api_key, img, model=model, strict_json=True)
    # 2차: 보조 모델 교차
    alt_model = "gpt-4o-mini" if model == "gpt-4o" else "gpt-4o"
    data2 = _ask_vision(api_key, img, model=alt_model, strict_json=True)

    data1 = _normalize_values(_normalize_keys(data1))
    data2 = _normalize_values(_normalize_keys(data2))

    merged = {}
    for k in FIELD_SCHEMA.keys():
        v1, v2 = data1.get(k), data2.get(k)
        if v1 == v2:
            merged[k] = v1
        else:
            pat = FIELD_SCHEMA[k].get("pattern")
            def ok(v): return bool(re.fullmatch(pat, str(v))) if pat and v is not None else False
            merged[k] = v1 if ok(v1) else (v2 if ok(v2) else (v1 or v2 or ""))

    merged = _validate_schema(merged)

    # 제목/첨부 누락 시 마지막 방어 재시도
    need_reask = False
    if (not merged.get("제목")) or (merged.get("attachment_count", 0) == 0):
        data3 = _ask_vision(api_key, img, model=model, strict_json=True)
        data3 = _normalize_values(_normalize_keys(data3))
        for k in ["제목", "attachment_count"]:
            if (not merged.get(k)) and data3.get(k):
                merged[k] = data3.get(k)
                need_reask = True
    if need_reask:
        merged = _validate_schema(merged)

    return merged


# -------------------- LLM 비교 (전체 출력 버전) --------------------
def compare_doc_with_guideline(
    api_key: str,
    doc_json: Dict[str, Any],
    guideline_chunks: List[str],
    model: str = "gpt-4o",
) -> str:
    llm = ChatOpenAI(model=model, temperature=0.0, api_key=api_key, max_tokens=2200)

    MAX_GUIDE = 12
    guideline_text = "\n\n".join(guideline_chunks[:MAX_GUIDE])
    user_doc_text = json.dumps(doc_json, ensure_ascii=False, indent=2)

    prompt = f"""
너는 회사 결재/경비 서류를 사전 검토하는 AI다.

[회사 가이드라인 및 유의사항 일부 (최대 {MAX_GUIDE}개)]
{guideline_text}

[사용자가 제출한 결재 서류(JSON)]
{user_doc_text}

요구사항:
1. 발견할 수 있는 모든 위반·누락·형식오류를 전부 나열해라.
2. 특히 다음은 반드시 체크:
   - '제목'이 비어 있거나 불완전한지
   - attachment_count가 0인데 문서 내용에 '출장','법인카드','개인카드','경비','지급요청','증빙','영수증' 등이 있는지
   - 필수 필드(지급요청일, 증빙유형, 카드내역 등)가 비어있는지
3. 결재선(결재/합의/승인/참조/수신)은 문제로 삼지 마라.
4. 출력 형식:

- 항목명: ...
- 문제점: ...
- 수정 예시: ...

- 항목명: ...
- 문제점: ...
- 수정 예시: ...
"""
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)


# ============================ UI ============================
with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )
    model = st.selectbox("GPT Vision / LLM 모델", ["gpt-4o-mini", "gpt-4o"], index=0)

col1, col2 = st.columns([1.1, 0.9])

# ------------ 왼쪽: 업로드 ------------
with col1:
    st.subheader("① 가이드라인 PDF 업로드")
    pdf_file = st.file_uploader("가이드라인 PDF", type=["pdf"], key="guide_pdf")
    if pdf_file is not None and st.button("가이드라인 임베딩 생성/업데이트"):
        if not api_key:
            st.error("먼저 API Key를 입력하세요.")
        else:
            raw = pdf_to_text(pdf_file.read())
            chunks = split_text(raw, chunk_size=800, overlap=120)
            embs = embed_texts(chunks, api_key)
            save_guideline_to_chroma(chunks, embs)
            st.session_state["guideline_ready"] = True

    st.subheader("② 유의사항 PDF 업로드")
    caution_pdf = st.file_uploader("유의사항 PDF", type=["pdf"], key="caution_pdf")
    if caution_pdf is not None and st.button("유의사항 임베딩 생성/업데이트"):
        if not api_key:
            st.error("먼저 API Key를 입력하세요.")
        else:
            raw = pdf_to_text(caution_pdf.read())
            chunks = split_text(raw, chunk_size=800, overlap=100)
            embs = embed_texts(chunks, api_key)
            save_caution_to_chroma(chunks, embs)
            st.session_state["caution_ready"] = True

    st.subheader("③ 결재/경비 서류 이미지 업로드")
    img_file = st.file_uploader("이미지 (jpg/png)", type=["jpg", "jpeg", "png"], key="doc_img")

    # 업로드 시 자동 인식
    if img_file is not None:
        img_bytes = img_file.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        st.image(Image.open(io.BytesIO(img_bytes)), caption="업로드한 결재 문서", use_container_width=True)

        need_run = st.session_state.get("last_img_hash") != img_hash or "doc_json" not in st.session_state

        if not api_key:
            st.warning("API Key가 필요합니다. 사이드바에 입력해 주세요.")
        elif need_run:
            with st.spinner("GPT가 문서 인식 중..."):
                doc_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                doc_json = gpt_extract_table(api_key, doc_img, model=model)
            st.session_state["doc_json"] = doc_json
            st.session_state["last_img_hash"] = img_hash
            st.success("문서 인식 완료 ✅")

        # 결과 JSON은 표시하지 않음 (요청 사항)
        if "doc_json" in st.session_state:
            pass  # 화면 출력 생략

# ------------ 오른쪽: 비교 ------------
with col2:
    st.subheader("④ 가이드라인 + 유의사항과 비교")
    if st.button("자동 검토 실행"):
        if not api_key:
            st.error("API Key가 필요합니다.")
        else:
            doc_json = st.session_state.get("doc_json")
            if not doc_json:
                st.error("먼저 결재 서류 이미지를 올리고 인식하세요.")
            else:
                guide_qs = [
                    "출장비용지급품의 작성 시 필수 항목은 무엇인가",
                    "지급요청일 입력 규칙은 무엇인가",
                    "증빙유형, 카드내역 입력 규칙은 무엇인가",
                ]
                caution_qs = [
                    "경비청구 시 주의해야 할 사항",
                    "허용되지 않는 비용과 예외 규정",
                ]
                attach_qs = [
                    "영수증, 카드전표, 첨부파일에 대한 규칙",
                    "법인카드 또는 개인카드 사용 시 필요한 첨부 서류",
                ]

                retrieved_texts: List[str] = []
                for q in guide_qs:
                    for r in search_guideline(q, api_key, k=3):
                        if r["metadata"].get("source") == "guideline":
                            retrieved_texts.append(r["text"])
                for q in caution_qs:
                    for r in search_guideline(q, api_key, k=3):
                        if r["metadata"].get("source") == "caution":
                            retrieved_texts.append(r["text"])
                for q in attach_qs:
                    for r in search_guideline(q, api_key, k=2):
                        retrieved_texts.append(r["text"])

                if not retrieved_texts:
                    st.error("가이드라인/유의사항이 없습니다. 먼저 PDF를 임베딩하세요.")
                else:
                    with st.spinner("가이드라인과 비교 중... (모든 위반사항을 찾는 중)"):
                        answer = compare_doc_with_guideline(api_key, doc_json, retrieved_texts, model=model)

                    st.success("검토 완료 ✅")
                    st.markdown("**검토 결과**")
                    st.write(answer)

                    payload = {
                        "doc_json": doc_json,
                        "retrieved_guideline_texts": retrieved_texts,
                        "analysis": answer,
                    }
                    st.download_button(
                        "검토 결과(JSON) 다운로드",
                        data=json.dumps(payload, ensure_ascii=False, indent=2),
                        file_name="guideline_check_result.json",
                        mime="application/json",
                    )
