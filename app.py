import streamlit as st
import os
import io
import base64
import json
import hashlib
from typing import List, Dict, Any
from PIL import Image
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


# -------------------- Vision: 이미지 → JSON --------------------
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def gpt_extract_table(api_key: str, img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    - 표 안 '제목' 우선, 없으면 상단 큰 제목
    - 결재선 무시
    - 첨부/증빙 관련 수치는 attachment_count
    """
    client = OpenAI(api_key=api_key)
    img_b64 = pil_to_b64(img)

    system_msg = (
        "너는 회사 결재/경비 문서를 표로 읽어 JSON으로 만드는 AI다. "
        "표 안 제목은 반드시 '제목'으로 넣고, 결재선은 무시하고, 첨부 개수는 'attachment_count'로 넣어라."
    )
    user_msg = (
        "아래 규칙으로 JSON을 만들어라.\n"
        "1) 표에 '제목' 셀/열이 있으면 그 값을 JSON의 '제목'으로.\n"
        "2) 없으면 상단 큰 제목을 '제목'으로.\n"
        "3) 결재/합의/승인/참조/수신 박스는 무시하거나 'approval_line_ignored': true만.\n"
        "4) '첨부','첨부 개수','증빙 개수','영수증 건수' 등은 숫자만 모아 'attachment_count'에.\n"
        "5) 나머지 표 셀도 key-value로 최대한 포함.\n"
        "6) JSON만 반환."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ],
            },
        ],
    )

    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json", "", 1).strip()

    try:
        data = json.loads(content)
    except Exception:
        data = {"_raw": content}

    if "제목" not in data:
        data["제목"] = ""
    if "attachment_count" not in data:
        data["attachment_count"] = 0
    return data


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
                doc_img = Image.open(io.BytesIO(img_bytes))
                doc_json = gpt_extract_table(api_key, doc_img, model=model)
            st.session_state["doc_json"] = doc_json
            st.session_state["last_img_hash"] = img_hash
            st.success("문서 인식 완료 ✅")

        if "doc_json" in st.session_state:
            st.code(json.dumps(st.session_state["doc_json"], ensure_ascii=False, indent=2), language="json")
            st.info(f"📎 인식된 첨부파일 개수: {st.session_state['doc_json'].get('attachment_count', 0)}")

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
