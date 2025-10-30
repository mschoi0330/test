import streamlit as st
import os, io, json, re, glob, hashlib, base64
from datetime import datetime
from typing import Dict, Any, List
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI

# ================== 기본 ==================
APP_TITLE = "📄 결재 서류 사전검토"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

APP_ROOT = os.getcwd()
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

# ================== 템플릿 레지스트리 ==================
TEMPLATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "지출결의서(AMARANTH 10)": {
        "schema_keys": [
            "제목","attachment_count","사용부서(팀)","사용자",
            "지급처","결의금액","지급요청일"
        ],
        "required_fields": [
            "제목","사용부서(팀)","사용자","지급처","지급요청일","결의금액"
        ],
        "key_alias": {
            "사용부서":"사용부서(팀)","부서":"사용부서(팀)",
            "제 목":"제목","제목 ":"제목",
            "합계":"결의금액","총합계":"결의금액",
            "첨부":"attachment_count","첨부파일수":"attachment_count","증빙개수":"attachment_count",
        },
        "folder": "expense_voucher",
    },
    "파견비신청서": {
        "schema_keys": [
            "프로젝트명","신청인","신청부서","신청기간","파견근무지",
            "일비(금액)","일비(산식)","일비(비고)",
            "교통비(금액)","교통비(산식)","교통비(비고)"
            ,"총합계","합계"
            "첨부(건수)","attachment_count","프로젝트코드"
        ],
        "required_fields": [
            "프로젝트명","신청인","신청부서","신청기간","파견근무지",
            "일비(금액)","일비(산식)","일비(비고)",
            "교통비(금액)","교통비(산식)","교통비(비고)"
            ,"총합계","합계"
            "첨부(건수)","attachment_count","프로젝트코드"        ],
        "key_alias": {
            "신청자":"신청인","신청부서(팀)":"신청부서","부서":"신청부서",
            "근무지":"파견근무지","파견지":"파견근무지",
            "일비 금액":"일비(금액)","일비 산식":"일비(산식)","일비 비고":"일비(비고)",
            "교통비 금액":"교통비(금액)","숙박비 금액":"숙박비(금액)","합계":"총합계",
            "첨부":"attachment_count","첨부파일수":"attachment_count","증빙개수":"attachment_count",
        },
        "folder": "dispatch_allowance",
    },
}
DEFAULT_TEMPLATE = "지출결의서"

# ================== 유틸 ==================
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    img = pil.convert("L")
    w, h = img.size
    scale = 2200 / max(w, h)
    if scale > 1.05:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 190 else (0 if p < 120 else p))
    return img.convert("RGB")

def detect_template_by_title(title: str) -> str:
    if not title:
        return DEFAULT_TEMPLATE
    t = title.strip()
    for name in TEMPLATE_REGISTRY.keys():
        if name in t:
            return name
    if any(k in t for k in ["파견", "파견비", "신청서"]):
        return "파견비신청서"
    return DEFAULT_TEMPLATE

def normalize_keys_template(d: dict, template: str) -> dict:
    aliases = TEMPLATE_REGISTRY.get(template, {}).get("key_alias", {})
    return {aliases.get(str(k).strip(), str(k).strip()): v for k, v in d.items()}

def ref_dirs_for_template(template: str):
    base = TEMPLATE_REGISTRY.get(template, {}).get("folder", "default")
    pass_dir = os.path.join(APP_ROOT, "data", "templates", base, "pass_json")
    fail_dir = os.path.join(APP_ROOT, "data", "templates", base, "fail_json")
    ensure_dir(pass_dir); ensure_dir(fail_dir)
    return pass_dir, fail_dir

# ================== Vision ==================
def ask_vision_values(api_key: str, pil_img: Image.Image, model: str, schema_keys: List[str]) -> dict:
    client = OpenAI(api_key=api_key)
    b64 = pil_to_b64(pil_img)
    sys = "설명 없이 JSON만. 지정된 키만 포함."
    usr = ("아래 표준키만 포함하는 JSON을 반환해. 키 목록:\n" + schema_keys.__str__() +
           "\n규칙: 1) 표의 '제목' 우선, 없으면 상단 큰 제목. 2) 결재/합의/승인/참조/수신 영역 무시. "
           "3) 'attachment_count' 또는 '첨부(건수)'는 숫자만, 없으면 0. 4) JSON만 출력.")
    resp = client.chat.completions.create(
        model=model, temperature=0, response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":[{"type":"text","text":usr},
                                            {"type":"image_url","image_url":{"url":b64}}]}]
    )
    return json.loads(resp.choices[0].message.content)

def gpt_extract_table(api_key: str, pil_img: Image.Image, model: str) -> dict:
    img = preprocess_for_ocr(pil_img)
    quick = ask_vision_values(api_key, img, model, ["제목"])
    title = (quick.get("제목") or "").strip()
    template = detect_template_by_title(title)
    schema_keys = TEMPLATE_REGISTRY.get(template, TEMPLATE_REGISTRY[DEFAULT_TEMPLATE])["schema_keys"]
    d1 = ask_vision_values(api_key, img, model, schema_keys)
    d1 = normalize_keys_template(d1, template)
    if "attachment_count" in d1:
        m = re.search(r"\d+", str(d1["attachment_count"])); d1["attachment_count"] = int(m.group()) if m else 0
    d1["__template__"] = template
    if "제목" not in d1: d1["제목"] = title
    return d1

# ================== 분석/빈칸 체크 ==================
def load_reference_stats_blank_only_for_template(template: str):
    pass_dir, fail_dir = ref_dirs_for_template(template)
    def _load_all(p):
        out=[]; 
        for f in glob.glob(os.path.join(p, "*.json")):
            try: out.append(json.load(open(f, "r", encoding="utf-8")))
            except: pass
        return out
    pass_docs, fail_docs = _load_all(pass_dir), _load_all(fail_dir)
    from collections import Counter
    cnt = Counter()
    for d in fail_docs:
        for k, v in d.items():
            if str(v).strip() == "": cnt[k] += 1
    return {"pass_count": len(pass_docs), "fail_count": len(fail_docs), "fail_empty_rank": cnt.most_common(10)}

def report_blanks_only(doc_json: dict, template: str):
    required = TEMPLATE_REGISTRY.get(template, {}).get("required_fields", [])
    issues = []
    for k in required:
        if str(doc_json.get(k, "")).strip() == "":
            issues.append({"항목명": k, "문제점": "빈칸", "수정 예시": f"{k} 값을 입력하세요."})
    return issues

def llm_explain_blanks(api_key: str, model: str, blanks: List[dict]) -> str:
    if not blanks: return ""
    client = OpenAI(api_key=api_key)
    prompt = ("다음 항목들이 빈칸입니다. 사용자가 바로 채울 수 있도록 간단·구체·한 줄 가이드로 요약해줘.\n" 
              + json.dumps(blanks, ensure_ascii=False))
    resp = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role":"system","content":"간결한 한국어 목록만 출력"},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

# ================== 사이드바 ==================
with st.sidebar:
    st.subheader("🔑 OpenAI 설정")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    model_vision = st.selectbox("Vision 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("---")
    st.info("※ 신뢰도 모드 / LLM 설명 생성 / 자동 템플릿 판별이 항상 활성화되어 있습니다.")

# 항상 활성화된 내부 설정
confidence_mode = True
llm_help_on = True
model_text = "gpt-4o"
manual_template = "자동판별"

# ================== 본문 ==================
col1, col2 = st.columns([1.1, 0.9])
with col1:
    st.subheader("① 이미지 업로드")
    img_file = st.file_uploader("결재/신청서 이미지 (jpg/png)", type=["jpg","jpeg","png"], key="doc_img")
    if img_file is not None:
        img_bytes = img_file.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        preview = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(preview, caption="업로드한 문서", use_container_width=True)

        if not api_key:
            st.warning("API Key를 입력하세요.")
        elif st.session_state.get("last_img_hash") != img_hash or "doc_json" not in st.session_state:
            with st.spinner("문서 인식 중..."):
                doc_json = gpt_extract_table(api_key, preview, model=model_vision)
            st.session_state["doc_json"] = doc_json
            st.session_state["last_img_hash"] = img_hash
            st.success(f"문서 인식 완료 ✅  (템플릿: {doc_json.get('__template__','?')})")

        if "doc_json" in st.session_state:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cur_template = st.session_state["doc_json"].get("__template__", DEFAULT_TEMPLATE)
            pass_dir, fail_dir = ref_dirs_for_template(cur_template)

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("PASS 샘플로 저장"):
                    path = os.path.join(pass_dir, f"pass_{ts}.json")
                    with open(path, "w", encoding="utf-8") as f: json.dump(st.session_state["doc_json"], f, ensure_ascii=False, indent=2)
                    st.success(f"[{cur_template}] PASS 저장: {path}")
            with b2:
                if st.button("FAIL 샘플로 저장"):
                    path = os.path.join(fail_dir, f"fail_{ts}.json")
                    with open(path, "w", encoding="utf-8") as f: json.dump(st.session_state["doc_json"], f, ensure_ascii=False, indent=2)
                    st.success(f"[{cur_template}] FAIL 저장: {path}")
            with b3:
                if st.button("이 문서 빈칸 확인"):
                    template = st.session_state["doc_json"].get("__template__", DEFAULT_TEMPLATE)
                    ref = load_reference_stats_blank_only_for_template(template)
                    blanks = report_blanks_only(st.session_state["doc_json"], template)
                    if not blanks:
                        st.success("빈칸 없음 ✅")
                    else:
                        st.error(f"빈칸 {len(blanks)}건 발견 ❌")
                        for it in blanks:
                            st.write(f"- **{it['항목명']}** → {it['수정 예시']}")
                        if llm_help_on:
                            with st.spinner("LLM 설명 생성 중..."):
                                tip = llm_explain_blanks(api_key, model_text, blanks)
                            st.markdown("**💡 LLM 권고문**")
                            st.write(tip)

with col2:
    st.subheader("② 템플릿/샘플 현황")
    cur_template = (st.session_state.get("doc_json") or {}).get("__template__", DEFAULT_TEMPLATE)
    config = TEMPLATE_REGISTRY.get(cur_template, {})
    st.write(f"- 현재 템플릿: **{cur_template}**")
    st.write(f"- 필수 필드: {', '.join(config.get('required_fields', []))}")
    st.write(f"- 전체 구조: {', '.join(config.get('schema_keys', []))}")
    ref = load_reference_stats_blank_only_for_template(cur_template)
    st.write(f"- PASS 샘플: {ref['pass_count']}개 / FAIL 샘플: {ref['fail_count']}개")
