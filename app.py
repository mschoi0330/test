import streamlit as st
import os
import io
import json
import base64
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -------------------------------------------------
# 설정
# -------------------------------------------------
APP_TITLE = "📄 AI 결재 사전검토 (RAG + PASS/FAIL 학습 통합형)"
DB_DIR = "./chroma_db"
DATASET_PATH = "./pass_fail_dataset.json"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# -------------------------------------------------
# 초기화
# -------------------------------------------------
chroma_client = chromadb.PersistentClient(path=DB_DIR)
GUIDE_COLLECTION_NAME = "company_guideline"

# -------------------------------------------------
# 공통 유틸
# -------------------------------------------------
def pdf_to_text(file: bytes) -> str:
    reader = PdfReader(io.BytesIO(file))
    texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(texts)

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

def embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    if not texts:
        return []
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    return embedder.embed_documents(texts)

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------------------------------------------
# PDF → Chroma 저장
# -------------------------------------------------
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
    texts = []
    for i in range(len(result["documents"][0])):
        texts.append(result["documents"][0][i])
    return texts

# -------------------------------------------------
# Vision: 결재 문서 분석
# -------------------------------------------------
def gpt_extract_table(api_key: str, img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    img_b64 = pil_to_b64(img)

    system_msg = (
        "너는 회사 결재/경비 문서를 표 형태로 분석하는 AI다. "
        "문서의 제목, 첨부파일 개수, 주요 항목(지급요청일, 증빙유형, 카드내역 등)을 JSON으로 만들어라. "
        "결재선(결재, 승인, 참조 등)은 무시하고, JSON만 반환해라."
    )
    user_msg = (
        "아래 결재서 이미지를 분석하여 JSON으로 만들어라.\n"
        "예시: { '제목': '출장비 결재서', '증빙유형': '법인카드', '첨부파일수': 2, '지급요청일': '익월 10일' }"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url", "image_url": {"url": img_b64}},
            ]},
        ],
    )

    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json", "", 1).strip()

    try:
        data = json.loads(content)
    except Exception:
        data = {"_raw": content}

    data.setdefault("제목", "")
    data.setdefault("첨부파일수", 0)
    return data

# -------------------------------------------------
# PASS / FAIL 데이터 학습 저장
# -------------------------------------------------
def save_pass_fail_data(new_data: List[Dict[str, Any]]):
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.extend(new_data)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    st.success(f"PASS/FAIL 데이터 {len(new_data)}건 저장 완료 (총 {len(existing)}건) ✅")

# -------------------------------------------------
# 통합 비교 (가이드라인 + 패턴 학습)
# -------------------------------------------------
def integrated_compare(api_key: str, test_json: Dict[str, Any], model: str = "gpt-4o") -> str:
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, max_tokens=2500)
    rag_texts = []

    # 가이드라인과 유의사항 검색
    for q in ["지급요청일 입력 규칙", "증빙유형 입력 규칙", "첨부파일 규칙", "경비청구 주의사항"]:
        rag_texts.extend(search_guideline(q, api_key, k=2))

    # PASS/FAIL 데이터 로드
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        pass_data = [d for d in dataset if d.get("상태") == "PASS"]
        fail_data = [d for d in dataset if d.get("상태") == "FAIL"]
    else:
        pass_data, fail_data = [], []

    prompt = f"""
너는 회사 결재 문서를 사전 검토하는 AI다.

[회사 가이드라인 및 유의사항 일부]
{json.dumps(rag_texts[:10], ensure_ascii=False, indent=2)}

[PASS 문서 예시]
{json.dumps(pass_data[:5], ensure_ascii=False, indent=2)}

[FAIL 문서 예시]
{json.dumps(fail_data[:5], ensure_ascii=False, indent=2)}

[검토할 결재 문서]
{json.dumps(test_json, ensure_ascii=False, indent=2)}

요구사항:
1. 가이드라인 규정 및 유의사항 위반 여부 확인.
2. PASS/FAIL 학습 데이터를 기반으로 테스트 문서의 패턴을 비교.
3. 필수 항목 누락(제목, 증빙유형, 첨부파일수, 지급요청일 등) 및 첨부파일수 0인 경우 FAIL 가능성 판단.
4. 출력은 아래 형식으로.

- 항목명: ...
- 문제점: ...
- 수정 예시: ...
최종 판정: PASS / FAIL (이유 포함)
"""
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)

# -------------------------------------------------
# UI
# -------------------------------------------------
with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("모델 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    st.caption("가이드라인 PDF, PASS/FAIL 학습 데이터를 업로드 후 테스트 문서를 검토하세요.")

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
                chunks = split_text(text)
                embs = embed_texts(chunks, api_key)
                save_pdf_to_chroma(chunks, embs, "guideline")
            if caution_pdf:
                text = pdf_to_text(caution_pdf.read())
                chunks = split_text(text)
                embs = embed_texts(chunks, api_key)
                save_pdf_to_chroma(chunks, embs, "caution")

    st.header("② PASS / FAIL 학습 데이터 업로드")
    pass_imgs = st.file_uploader("✅ PASS 문서 (여러장 가능)", type=["jpg", "png"], accept_multiple_files=True)
    fail_imgs = st.file_uploader("❌ FAIL 문서 (여러장 가능)", type=["jpg", "png"], accept_multiple_files=True)

    if st.button("PASS/FAIL 데이터 학습"):
        if not api_key:
            st.error("API Key가 필요합니다.")
        else:
            learned = []
            for img_file in pass_imgs or []:
                img = Image.open(img_file)
                data = gpt_extract_table(api_key, img, model)
                data["상태"] = "PASS"
                learned.append(data)
            for img_file in fail_imgs or []:
                img = Image.open(img_file)
                data = gpt_extract_table(api_key, img, model)
                data["상태"] = "FAIL"
                learned.append(data)
            if learned:
                save_pass_fail_data(learned)

    st.header("③ 테스트 결재 문서 업로드")
    test_img = st.file_uploader("테스트 문서", type=["jpg", "png"], key="test_img")
    if test_img:
        img = Image.open(test_img)
        st.image(img, caption="테스트 문서", use_container_width=True)
        if st.button("문서 분석 실행"):
            test_json = gpt_extract_table(api_key, img, model)
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
