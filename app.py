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
    # source 기준으로만 삭제
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
        docs.append(
            {
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
            }
        )
    return docs


# -------------------- Vision: 이미지 → JSON --------------------
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def gpt_extract_table(api_key: str, img: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    - 표 안 '제목' 우선
    - 없으면 상단 큰 제목
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
        "1. 표(테이블) 안에 '제목'이라는 셀/행/열이 있으면 그 값을 JSON의 '제목'으로 넣어. (1순위)\n"
        "2. 표 안에 '제목'이 없으면 문서 상단의 큰 제목을 '제목'으로 넣어.\n"
        "3. 결재/합의/승인/참조/수신/기안자 이름이 있는 박스는 무시하거나 'approval_line_ignored': true 로만 넣어.\n"
        "4. '첨부', '첨부파일', '첨부 개수', '증빙 개수', '영수증 건수'처럼 보이는 칸이 있으면 그 칸의 숫자를 모아서 "
        "JSON의 'attachment_count' 키에 넣어. 숫자가 안 보이면 0으로 넣어.\n"
        "5. 다른 표 셀(회사, 사용부서, 사용자, 지급처, 업무추진비, 결의금액, 지급요청일 등)은 가능한 한 key-value로 넣어.\n"
        "6. JSON만 반환해."
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
    """
    - 가져온 가이드라인 전부 보고
    - 제목 누락
    - 첨부파일 미첨부
    - 기타 가이드/유의사항 위반
    을 전부 나열
    """
    clean_doc_json = {k: v for k, v in doc_json.items() if k != "approval_line_ignored"}

    # ❗ 여기서 max_tokens를 크게 줘서 중간에 안 잘리게 한다
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=api_key,
        max_tokens=2200,   # 필요하면 더 늘려도 됨
    )

    # 너무 많이 넣으면 자꾸 잘려서 상위 12개만
    MAX_GUIDE = 12
    guideline_text = "\n\n".join(guideline_chunks[:MAX_GUIDE])
    user_doc_text = json.dumps(doc_json, ensure_ascii=False, indent=2)

    prompt = f"""
너는 회사 결재/경비 서류를 사전 검토하는 AI다.

아래는 회사에서 제공한 가이드라인/유의사항 일부다. 중요한 부분만 골라서 쓰면 된다.

[회사 가이드라인 및 유의사항 일부 (최대 {MAX_GUIDE}개)]
{guideline_text}

[사용자가 제출한 결재 서류(JSON)]
{user_doc_text}

요구사항:
1. 발견할 수 있는 **모든** 위반·누락·형식오류를 전부 나열해라. 하나만 쓰고 끝내지 마라.
2. 특히 다음은 반드시 체크해라:
   - 표 안의 '제목'이 비어 있거나 불완전한지
   - attachment_count가 0인데 문서 내용에 '출장', '법인카드', '개인카드', '경비', '지급요청', '증빙', '영수증' 등이 있는지
   - 가이드라인에서 필수라고 한 필드(예: 지급요청일, 증빙유형, 카드내역 등)가 JSON에서 비어 있는지
3. 결재선(결재/합의/승인/참조/수신)은 문제로 삼지 마라.
4. 출력 형식은 아래 형식으로만 써라. 여러 개면 여러 개를 이어서 써라.

- 항목명: ...
- 문제점: ...
- 수정 예시: ...

- 항목명: ...
- 문제점: ...
- 수정 예시: ...

'가이드라인 근거:', '출처:' 같은 문장은 쓰지 마라.
"""
    res = llm.invoke(prompt)
    return res.content if hasattr(res, "content") else str(res)



# ------------------------------------------------------------------------------
# 8. Streamlit UI 구성
# ------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )
    model = st.selectbox("GPT Vision / LLM 모델", ["gpt-4o-mini", "gpt-4o"], index=0)

col1, col2 = st.columns([1.1, 0.9])


# ------------------------------------------------------------------------------
# 본문 레이아웃
# ------------------------------------------------------------------------------
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

    # 업로드 파일이 바뀌면 자동 인식 (버튼 없이)
    if img_file is not None:
        img_bytes = img_file.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # 미리보기
        st.image(Image.open(io.BytesIO(img_bytes)), caption="업로드한 결재 문서", use_container_width=True)

        # 파일이 새로 올라왔거나 다른 파일이면 자동 인식
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

        # 인식 결과 표시
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
            # 결재 서류가 있는지 확인
            doc_json = st.session_state.get("doc_json")
            if not doc_json:
                st.error("먼저 결재 서류 이미지를 올리고 인식하세요.")
            else:
                # RAG: 필요한 쿼리들
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
                        # 첨부 규칙은 가이드/유의사항 어느 쪽이든 다 가져옴
                        retrieved_texts.append(r["text"])

                if not retrieved_texts:
                    st.error("가이드라인/유의사항이 없습니다. 먼저 PDF를 임베딩하세요.")
                else:
                    with st.spinner("가이드라인과 비교 중... (모든 위반사항을 찾는 중)"):
                        answer = compare_doc_with_guideline(
                            api_key, doc_json, retrieved_texts, model=model
                        )

                    st.success("검토 완료 ✅")
                    st.markdown("**검토 결과**")
                    st.write(answer)

                    # 결과 다운로드
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
