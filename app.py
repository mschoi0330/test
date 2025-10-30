# app.py — 템플릿 분기 + PASS/FAIL 레퍼런스 기반 "빈칸만" 점검 + LLM 보조(신뢰도/설명)
import streamlit as st
import os, io, json, re, glob, hashlib, base64
from datetime import datetime
from typing import Dict, Any, List

from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI

# ================== 앱 기본 ==================
APP_TITLE = "📄 결재 서류 빈칸 점검 (템플릿 분기 + PASS/FAIL + LLM 보조)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

APP_ROOT = os.getcwd()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ================== 템플릿 레지스트리 ==================
# 필요 시 여기만 추가하면 템플릿 확장 가능
TEMPLATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "지출결의서": {
        "schema_keys": [
            "제목","attachment_count","회사","사용부서(팀)","사용자",
            "지급처","업무추진비","결의금액","지급요청일"
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
            "제목","신청인","신청부서","신청기간","파견근무지",
            "일비(금액)","일비(산식)","일비(비고)",
            "교통비(금액)","숙박비(금액)","총합계",
            "첨부(건수)","attachment_count"
        ],
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
    """저해상도 스크린샷 대응: 확대+대비/샤픈+간이 이진화"""
    img = pil.convert("L")
    w, h = img.size
    scale = 2200 / max(w, h)  # 1600 -> 2200로 상향
    if scale > 1.05:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 190 else (0 if p < 120 else p))
    return img.convert("RGB")

# ================== 템플릿 보조 함수 ==================
def detect_template_by_title(title: str) -> str:
    if not title: 
        return DEFAULT_TEMPLATE
    t = title.strip()
    for name in TEMPLATE_REGISTRY.keys():
        if name in t:
            return name
    # 키워드 휴리스틱
    if any(k in t for k in ["파견", "파견비", "신청서"]):
        return "파견비신청서"
    return DEFAULT_TEMPLATE

def normalize_keys_template(d: dict, template_name: str) -> dict:
    aliases = TEMPLATE_REGISTRY.get(template_name, {}).get("key_alias", {})
    out = {}
    for k, v in d.items():
        k2 = aliases.get(str(k).strip(), str(k).strip())
        out[k2] = v
    return out

def ref_dirs_for_template(template: str):
    base = TEMPLATE_REGISTRY.get(template, {}).get("folder", "default")
    pass_dir = os.path.join(APP_ROOT, "data", "templates", base, "pass_json")
    fail_dir = os.path.join(APP_ROOT, "data", "templates", base, "fail_json")
    ensure_dir(pass_dir); ensure_dir(fail_dir)
    return pass_dir, fail_dir

# ================== Vision 호출 ==================
def ask_vision_values(api_key: str, pil_img: Image.Image, model: str, schema_keys: List[str]) -> dict:
    client = OpenAI(api_key=api_key)
    b64 = pil_to_b64(pil_img)
    sys = "설명 없이 JSON만 출력. 지정된 키만 포함."
    usr = (
        "아래 표준키만 포함하는 JSON을 반환해. 키 목록:\n"
        + schema_keys.__str__() +
        "\n규칙:\n"
        "1) 표에 '제목' 셀이 있으면 그 값을 '제목'에. 없으면 상단 큰 제목을 사용.\n"
        "2) 결재/합의/승인/참조/수신 영역은 무시 가능.\n"
        "3) 'attachment_count' 또는 '첨부(건수)'는 숫자만. 없으면 0.\n"
        "4) JSON만 출력."
    )
    resp = client.chat.completions.create(
        model=model, temperature=0, response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":[{"type":"text","text":usr},
                                            {"type":"image_url","image_url":{"url":b64}}]}]
    )
    return json.loads(resp.choices[0].message.content)

def ask_vision_with_conf(api_key: str, pil_img: Image.Image, model: str, schema_keys: List[str]) -> dict:
    client = OpenAI(api_key=api_key)
    b64 = pil_to_b64(pil_img)
    sys = "설명 없이 JSON만. 각 키는 {'value':..., 'confidence':0~1} 형태."
    usr = ("표준키만 포함:\n" + schema_keys.__str__() +
           "\n각 항목은 {'value':<문자열>, 'confidence':<0~1 float>}.\n"
           "규칙: '제목' 우선, 결재선 무시, 첨부/건수는 숫자.")
    resp = client.chat.completions.create(
        model=model, temperature=0, response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":[{"type":"text","text":usr},
                                            {"type":"image_url","image_url":{"url":b64}}]}]
    )
    return json.loads(resp.choices[0].message.content)

# ================== 메인 추출 파이프라인 ==================
def gpt_extract_table(api_key: str, pil_img: Image.Image, model: str, use_confidence: bool) -> dict:
    img = preprocess_for_ocr(pil_img)

    # 0) 제목만 먼저 빠르게 뽑아 템플릿 추정
    quick = ask_vision_values(api_key, img, model, schema_keys=["제목"])
    title = (quick.get("제목") or "").strip()
    template = detect_template_by_title(title)
    schema_keys = TEMPLATE_REGISTRY.get(template, TEMPLATE_REGISTRY[DEFAULT_TEMPLATE])["schema_keys"]

    # 1차
    d1 = (ask_vision_with_conf if use_confidence else ask_vision_values)(api_key, img, model, schema_keys)
    if use_confidence:
        d1 = {k:(v.get("value") if isinstance(v,dict) else v) for k,v in d1.items()}
    d1 = normalize_keys_template(d1, template)

    # 2차(교차 모델)
    alt = "gpt-4o" if model=="gpt-4o-mini" else "gpt-4o-mini"
    d2 = (ask_vision_with_conf if use_confidence else ask_vision_values)(api_key, img, alt, schema_keys)
    if use_confidence:
        d2 = {k:(v.get("value") if isinstance(v,dict) else v) for k,v in d2.items()}
    d2 = normalize_keys_template(d2, template)

    # 간단 병합(값 있으면 채택)
    merged = {}
    for k in schema_keys:
        v1, v2 = d1.get(k), d2.get(k)
        merged[k] = v1 if str(v1).strip() else (v2 if str(v2).strip() else "")

    # 공통 보정
    if "attachment_count" in merged:
        m = re.search(r"\d+", str(merged["attachment_count"]))
        merged["attachment_count"] = int(m.group()) if m else 0
    if "첨부(건수)" in merged and not merged.get("attachment_count"):
        m = re.search(r"\d+", str(merged["첨부(건수)"]))
        merged["attachment_count"] = int(m.group()) if m else 0

    merged["__template__"] = template
    if "제목" not in merged: merged["제목"] = title
    return merged

# ================== 레퍼런스(빈칸 전용) ==================
def load_reference_stats_blank_only_for_template(template: str, required_threshold=0.8):
    pass_dir, fail_dir = ref_dirs_for_template(template)

    def _load_all(p):
        out=[]
        for f in glob.glob(os.path.join(p, "*.json")):
            try: out.append(json.load(open(f, "r", encoding="utf-8")))
            except: pass
        return out

    pass_docs = _load_all(pass_dir)
    fail_docs = _load_all(fail_dir)

    filled_ratio = {}
    if pass_docs:
        keys = set().union(*[d.keys() for d in pass_docs])
        for k in keys:
            vals = [(str(d.get(k, "")).strip() != "") for d in pass_docs]
            if vals:
                filled_ratio[k] = sum(vals) / len(vals)

    inferred_required = {k for k, r in filled_ratio.items() if r >= required_threshold}

    from collections import Counter
    fail_empty_rank = []
    if fail_docs:
        cnt = Counter()
        for d in fail_docs:
            for k, v in d.items():
                if str(v).strip() == "":
                    cnt[k] += 1
        fail_empty_rank = cnt.most_common(10)

    return {
        "inferred_required": inferred_required,
        "pass_count": len(pass_docs),
        "fail_count": len(fail_docs),
        "filled_ratio": filled_ratio,
        "fail_empty_rank": fail_empty_rank,
        "pass_dir": pass_dir,
        "fail_dir": fail_dir,
    }

def report_blanks_only(doc_json: dict, ref_stats: dict):
    req = ref_stats.get("inferred_required", set())
    issues = []
    for k in sorted(req):
        if str(doc_json.get(k, "")).strip() == "":
            issues.append({"항목명": k, "문제점": "빈칸", "수정 예시": f"{k} 값을 입력하세요."})
    return issues

# ================== LLM 설명(옵션) ==================
def llm_explain_required(api_key: str, model: str, filled_ratio: dict, topk=5) -> str:
    if not filled_ratio: return ""
    client = OpenAI(api_key=api_key)
    top = sorted(filled_ratio.items(), key=lambda x: -x[1])[:topk]
    prompt = (
        "다음 채움율 목록을 바탕으로 왜 해당 필드들을 '사실상 필수'로 보는지 한 줄 요약을 작성해줘. "
        "실무자가 이해하기 쉽게, 한국어, 간결하게.\n" + json.dumps(top, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role":"system","content":"간결한 한국어 설명만 출력"},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

def llm_explain_blanks(api_key: str, model: str, blanks: List[dict]) -> str:
    if not blanks: return ""
    client = OpenAI(api_key=api_key)
    prompt = (
        "다음 항목들이 빈칸입니다. 사용자가 바로 채울 수 있도록 간단·구체·한 줄 가이드로 요약해줘. "
        "불필요한 서론 없이 목록 형태로.\n" + json.dumps(blanks, ensure_ascii=False)
    )
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
    use_confidence = st.toggle("신뢰도(Confidence) 모드 사용", value=False)
    st.markdown("---")
    llm_help_on = st.toggle("LLM 설명 생성(권고문/이유) 사용", value=False)
    model_text = st.selectbox("설명용 LLM", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=not llm_help_on)
    st.markdown("---")
    # 템플릿 선택(수동 오버라이드용)
    manual_template = st.selectbox("템플릿(수동 선택, 자동판별 무시)", ["자동판별"] + list(TEMPLATE_REGISTRY.keys()), index=0)

# ================== 본문 레이아웃 ==================
col1, col2 = st.columns([1.1, 0.9])

# -------- 왼쪽: 업로드/저장/확인(버튼 3개) --------
with col1:
    st.subheader("① 결재/경비 서류 이미지 업로드")
    img_file = st.file_uploader("이미지 (jpg/png)", type=["jpg","jpeg","png"], key="doc_img")

    if img_file is not None:
        img_bytes = img_file.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        preview = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(preview, caption="업로드한 결재 문서", use_container_width=True)

        need_run = st.session_state.get("last_img_hash") != img_hash or "doc_json" not in st.session_state
        if not api_key:
            st.warning("API Key가 필요합니다. 사이드바에 입력해 주세요.")
        elif need_run:
            with st.spinner("문서 인식 중..."):
                doc_json = gpt_extract_table(api_key, preview, model=model_vision, use_confidence=use_confidence)
                # 수동 템플릿 지정 시 교체
                if manual_template != "자동판별":
                    doc_json["__template__"] = manual_template
            st.session_state["doc_json"] = doc_json
            st.session_state["last_img_hash"] = img_hash
            st.success(f"문서 인식 완료 ✅  (템플릿: {st.session_state['doc_json'].get('__template__','?')})")

        # === 버튼 3개 ===
        if "doc_json" in st.session_state:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cur_template = st.session_state["doc_json"].get("__template__", DEFAULT_TEMPLATE)
            pass_dir, fail_dir = ref_dirs_for_template(cur_template)

            b1, b2, b3 = st.columns(3)

            with b1:
                if st.button("PASS 샘플로 저장"):
                    ensure_dir(pass_dir)
                    path = os.path.join(pass_dir, f"pass_{ts}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state["doc_json"], f, ensure_ascii=False, indent=2)
                    st.success(f"[{cur_template}] PASS 샘플 저장 완료: {path}")

            with b2:
                if st.button("FAIL 샘플로 저장"):
                    ensure_dir(fail_dir)
                    path = os.path.join(fail_dir, f"fail_{ts}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state["doc_json"], f, ensure_ascii=False, indent=2)
                    st.success(f"[{cur_template}] FAIL 샘플 저장 완료: {path}")

            with b3:
                if st.button("이 문서 빈칸 확인"):
                    template = st.session_state["doc_json"].get("__template__", DEFAULT_TEMPLATE)
                    ref = load_reference_stats_blank_only_for_template(template)
                    if ref.get("pass_count", 0) + ref.get("fail_count", 0) == 0:
                        st.warning(f"[{template}] 레퍼런스가 없습니다. PASS/FAIL 샘플을 저장해 주세요.")
                    else:
                        blanks = report_blanks_only(st.session_state["doc_json"], ref)
                        if not blanks:
                            st.success("빈칸 없음 ✅ (템플릿 레퍼런스 기준 필수필드 모두 채워짐)")
                        else:
                            st.error(f"빈칸 {len(blanks)}건 발견 ❌")
                            with st.expander("빈칸 상세 보기", expanded=True):
                                for it in blanks:
                                    st.write(f"- **항목명**: {it['항목명']}\n  - 문제점: {it['문제점']}\n  - 수정 예시: {it['수정 예시']}")
                            if llm_help_on and api_key:
                                with st.spinner("LLM이 간단 가이드를 작성 중..."):
                                    advice = llm_explain_blanks(api_key, model_text, blanks)
                                if advice:
                                    st.markdown("**🔎 입력 가이드 (LLM 요약)**")
                                    st.write(advice)

# -------- 오른쪽: 레퍼런스 현황 / 필수 필드 설명 --------
with col2:
    st.subheader("② 레퍼런스 현황 / 필수 필드 설명")
    # 현재 문서 템플릿 기준 현황 표시
    cur_template = (st.session_state.get("doc_json") or {}).get("__template__", DEFAULT_TEMPLATE)
    ref = load_reference_stats_blank_only_for_template(cur_template)
    st.write(f"- 현재 템플릿: **{cur_template}**")
    st.write(f"- PASS 샘플: **{ref['pass_count']}**개, FAIL 샘플: **{ref['fail_count']}**개")
    st.write(f"- 추론된 '사실상 필수' 필드 수: **{len(ref['inferred_required'])}**개")
    if ref["inferred_required"]:
        st.write("필수로 간주된 필드 예:", ", ".join(list(ref["inferred_required"])[:10]) + (" ..." if len(ref["inferred_required"])>10 else ""))

    if ref["fail_empty_rank"]:
        st.caption("FAIL에서 자주 비던 필드 Top: " + ", ".join([f"{k}({c})" for k,c in ref["fail_empty_rank"]]))

    if llm_help_on and api_key and ref.get("filled_ratio"):
        if st.button("필수로 보는 이유 한 줄 설명(LLM)"):
            with st.spinner("LLM이 요약 설명 작성 중..."):
                msg = llm_explain_required(api_key, model_text, ref["filled_ratio"], topk=5)
            if msg:
                st.markdown("**📝 왜 필수로 보나요? (LLM 요약)**")
                st.write(msg)
