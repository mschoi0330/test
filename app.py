import streamlit as st
import os
import io
import base64
import json
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

APP_TITLE = "📄 AI 결재 사전검토 (제목+첨부파일 인식 자동판단)"

DB_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_DIR)
GUIDE_COLLECTION_NAME = "company_guideline"


# -------------------- 공통 유틸 --------------------
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
    vectors = embedder.embed_documents(texts)
    return vectors


# -------------------- Chroma 저장 --------------------
def save_guideline_to_chroma(chunks: List[str], embeddings: List[List[float]]):
    if not chunks or not embeddings:
        st.error("가이드라인 PDF에서 텍스트를 찾지 못했습니다. (스캔본이거나 빈 문서일 수 있어요)")
        return
    collection = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    collection.delete(where={"source": "guideline"})
    ids = [f"guide_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "guideline", "chunk": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    st.success(f"가이드라인 {len(chunks)}개 청크 저장 완료 ✅")


def save_caution_to_chroma(chunks: List[str], embeddings: List[List[float]]):
    if not chunks or not embeddings:
        st.error("유의사항 PDF에서 텍스트를 찾지 못했습니다.")
        return
    collection = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    collection.delete(where={"source": "caution"})
    base_idx = 10_000
    ids = [f"caution_{base_idx + i}" for i in range(len(chunks))]
    metadatas = [{"source": "caution", "chunk": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    st.success(f"유의사항 {len(chunks)}개 청크 저장 완료 ✅")


def search_guideline(query: str, api_key: str, k: int = 4) -> List[Dict[str, Any]]:
    collection = chroma_client.get_or_create_collection(GUIDE_COLLECTION_NAME)
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    q_emb = embedder.embed_query(query)
    result = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for i in range(len(result["documents"][0])):
        docs.append({"text": result["documents"][0][i], "metadata": result["metadatas"][0][i]})
    return docs


# -------------------- GPT Vision --------------------
def pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def gpt_extract_table(api_key: str, pil_img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    - 표 안의 '제목' 셀을 최우선으로 인식
    - 상단 큰 제목은 fallback
    - 결재선은 무시
    - '첨부'나 '증빙 개수' 관련 칸이 있으면 숫자를 attachment_count로 추출
    """
    client = OpenAI(api_key=api_key)
    img_b64 = pil_to_b64(pil_img)

    system_msg = (
        "너는 회사 결재/경비 문서를 표로 읽어 JSON으로 만드는 AI다. "
        "표 안 제목은 반드시 '제목' 키로, 첨부 개수는 'attachment_count'로 뽑아라. "
        "결재선(결재/합의/승인/참조/수신)은 무시한다."
    )
    user_msg = (
        "다음 규칙으로 JSON을 만들어:\n"
        "1. 표 안에 '제목'이라는 셀 이름이 있으면 그 값을 JSON의 '제목'으로 넣어.\n"
        "2. 표 안에 '제목'이 없으면 문서 상단의 큰 제목(예: 출장비용지급품의)을 '제목'으로 넣어.\n"
        "3. 결재/합의/승인/참조/수신 칸은 무시하거나 'approval_line_ignored': true로만 남겨.\n"
        "4. '첨부', '첨부파일', '첨부 개수', '증빙 개수', '영수증 건수' 같은 칸이 있으면 "
        "숫자만 뽑아서 'attachment_count': <숫자> 로 넣어. 숫자가 없으면 0으로.\n"
        "5. 표에 있는 나머지 값들도 key-value로 최대한 넣어.\n"
        "6. JSON만 출력해."
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


# -------------------- LLM 비교 --------------------
def compare_doc_with_guideline(api_key: str, doc_json: Dict[str, Any], guideline_chunks: List[str], model: str = "gpt-4o") -> str:
    llm = ChatOpenAI(model=model, temperature=0.0, api_key=api_key)
    guideline_text = "\n\n".join(guideline_chunks)
    user_doc_text = json.dumps(doc_json, ensure_ascii=False, indent=2)

    prompt = f"""
너는 회사 결재/경비 서류를 사전 검토하는 AI야.

[회사 가이드라인 및 유의사항 내용]
{guideline_text}

[사용자 문서(JSON)]
{user_doc_text}

검토 기준:
1. '제목' 필드는 반드시 존재해야 한다. 비어 있으면 '제목 누락'으로 지적하라.
2. attachment_count가 0인데 문서 내용에 '출장', '법인카드', '개인카드', '증빙', '영수증', '지급요청' 등이 포함되어 있으면 '증빙 첨부 누락'으로 지적하라.
3. 결재선(결재/합의/승인/참조/수신)은 무시한다.
4. 결과는 아래 형식으로만 출력하라:

- 항목명: ...
- 문제점: ...
- 수정 예시: ...

'가이드라인 근거' 같은 문구는 쓰지 마.
"""
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("표 제목·첨부파일 자동 인식 + 가이드라인/유의사항 기반 검토")

with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    model = st.selectbox("GPT Vision / LLM 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("---")
    st.info("가이드라인과 유의사항 PDF를 업로드하고 임베딩을 생성하세요.")

col1, col2 = st.columns([1.1, 0.9])

# 왼쪽
with col1:
    st.subheader("① 가이드라인 PDF 업로드")
    pdf_file = st.file_uploader("가이드라인 PDF", type=["pdf"], key="guide_pdf")
    if pdf_file is not None and st.button("가이드라인 임베딩 생성/업데이트"):
        if not api_key:
            st.error("먼저 OpenAI API Key를 입력하세요.")
        else:
            raw_text = pdf_to_text(pdf_file.read())
            chunks = split_text(raw_text, chunk_size=800, overlap=120)
            embeddings = embed_texts(chunks, api_key)
            save_guideline_to_chroma(chunks, embeddings)
            st.session_state["guideline_ready"] = True

    st.subheader("② 유의사항 PDF 업로드")
    caution_pdf = st.file_uploader("유의사항 PDF", type=["pdf"], key="caution_pdf")
    if caution_pdf is not None and st.button("유의사항 임베딩 생성/업데이트"):
        raw_text = pdf_to_text(caution_pdf.read())
        chunks = split_text(raw_text, chunk_size=800, overlap=100)
        embeddings = embed_texts(chunks, api_key)
        save_caution_to_chroma(chunks, embeddings)
        st.session_state["caution_ready"] = True

    st.subheader("③ 결재 문서 이미지 업로드")
    img_file = st.file_uploader("결재 서류 이미지 (jpg/png)", type=["jpg", "jpeg", "png"], key="doc_img")
    if img_file is not None:
        doc_img = Image.open(img_file)
        st.image(doc_img, caption="업로드한 결재 문서", use_column_width=True)
        if st.button("이미지에서 표/제목/첨부 인식", type="primary"):
            with st.spinner("GPT가 문서 분석 중..."):
                doc_json = gpt_extract_table(api_key, doc_img, model=model)
            st.session_state["doc_json"] = doc_json
            st.success("문서 분석 완료 ✅")
            st.code(json.dumps(doc_json, ensure_ascii=False, indent=2), language="json")
            if "attachment_count" in doc_json:
                st.info(f"인식된 첨부파일 개수: {doc_json['attachment_count']}")

# 오른쪽
with col2:
    st.subheader("④ 가이드라인 + 유의사항 비교")
    if st.button("자동 검토 실행"):
        doc_json = st.session_state.get("doc_json")
        if not doc_json:
            st.error("먼저 결재 문서를 업로드하고 분석하세요.")
        else:
            guide_queries = [
                "출장비용지급품의 작성 시 필수 항목은 무엇인가",
                "법인카드/개인카드 사용 시 증빙 첨부 규칙은 무엇인가",
                "지급요청일 입력 규칙은 무엇인가",
            ]
            caution_queries = ["경비청구 시 주의사항", "허용되지 않는 비용 항목"]

            docs: List[str] = []
            for q in guide_queries + caution_queries:
                found = search_guideline(q, api_key, k=2)
                for f in found:
                    docs.append(f["text"])

            with st.spinner("가이드라인/유의사항 기반 검토 중..."):
                answer = compare_doc_with_guideline(api_key, doc_json, docs, model=model)

            st.success("검토 완료 ✅")
            st.markdown("**검토 결과**")
            st.write(answer)

            st.download_button(
                "검토 결과(JSON) 다운로드",
                data=json.dumps(
                    {"doc_json": doc_json, "guideline_texts": docs, "analysis": answer},
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="guideline_check_result.json",
                mime="application/json",
            )
