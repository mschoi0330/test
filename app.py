# app.py — PASS/FAIL 레퍼런스 기반 "빈칸만" 점검 + LLM 보조(신뢰도/설명)
import streamlit as st
import os, io, json, re, glob, hashlib, base64
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from openai import OpenAI

# ================== 앱 기본 ==================
APP_TITLE = "📄 결재 서류 빈칸 점검 (PASS/FAIL 레퍼런스 + LLM 보조)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

APP_ROOT = os.getcwd()
DATA_DIR = os.path.join(APP_ROOT, "data")
PASS_DIR = os.path.join(DATA_DIR, "pass_json")
FAIL_DIR = os.path.join(DATA_DIR, "fail_json")

def ensure_dirs():
    os.makedirs(PASS_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)

ensure_dirs()

# ================== 유틸 ==================
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# ================== Vision 전처리/인식 ==================
def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    img = pil.convert("L")
    w, h = img.size
    scale = 1600 / max(w, h)
    if scale > 1.05:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 190 else (0 if p < 120 else p))
    return img.convert("RGB")

# 표준 스키마(필요시 키 추가)
FIELD_SCHEMA = {
    "제목": {"required": True, "pattern": r".{2,}"},
    "attachment_count": {"required": True, "pattern": r"^\d+$"},
    "회사": {"required": False, "pattern": r".*"},
    "사용부서(팀)": {"required": False, "pattern": r".*"},
    "사용자": {"required": False, "pattern": r".*"},
    "지급처": {"required": False, "pattern": r".*"},
    "업무추신비": {"required": False, "pattern": r"^\d{1,3}(,\d{3})*$"},  # 오탈자 대비용 예시 키 추가 가능
    "업무추진비": {"required": False, "pattern": r"^\d{1,3}(,\d{3})*$"},
    "결의금액": {"required": False, "pattern": r"^\d{1,3}(,\d{3})*$"},
    "지급요청일": {"required": False, "pattern": r"^\d{4}-\d{2}-\d{2}(\([^)]+\))?$"},
}

# 라벨 정규화(동의어/오탈자 → 표준키)
KEY_NORMALIZER = {
    "사용부서": "사용부서(팀)",
    "부서": "사용부서(팀)",
    "제 목": "제목",
    "제목 ": "제목",
    "합계": "결의금액",
    "총합계": "결의금액",
    "업무추신비": "업무추진비",
}

def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        k2 = KEY_NORMALIZER.get(str(k).strip(), str(k).strip())
        out[k2] = v
    return out

def _normalize_values(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    def norm_money(s):
        s = str(s).strip()
        s = re.sub(r"[^\d,]", "", s)
        if s.isdigit():
            s = f"{int(s):,}"
        return s
    if "업무추진비" in out:
        out["업무추진비"] = norm_money(out["업무추진비"])
    if "결의금액" in out:
        out["결의금액"] = norm_money(out["결의금액"])
    if "지급요청일" in out:
        s = str(out["지급요청일"])
        s2 = re.sub(r"[./]", "-", s)
        m = re.search(r"(\d{4}-\d{2}-\d{2})(\([^)]+\))?", s2)
        if m:
            out["지급요청일"] = m.group(0)
    if "attachment_count" in out:
        m = re.search(r"\d+", str(out["attachment_count"]))
        out["attachment_count"] = int(m.group()) if m else 0
    return out

# ===== Vision 호출: (1) 값만 / (2) 신뢰도 포함 =====
def ask_vision_values(api_key: str, pil_img: Image.Image, model: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    b64 = pil_to_b64(pil_img)
    sys = "설명 없이 JSON만 출력. 지정된 키만 포함."
    usr = (
        "아래 표준키만 포함하는 JSON을 반환해. 키 목록:\n"
        + list(FIELD_SCHEMA.keys()).__str__()
        + "\n규칙:\n"
        "1) 표에 '제목' 셀이 있으면 그 값을 '제목'에. 없으면 상단 큰 제목을 사용.\n"
        "2) 결재/합의/승인/참조/수신 영역은 무시 가능.\n"
        "3) 'attachment_count'는 숫자만. 없으면 0.\n"
        "4) '업무추진비','결의금액'은 숫자와 콤마만(예: 150,000).\n"
        "5) '지급요청일'은 YYYY-MM-DD 또는 YYYY-MM-DD(요일) 형식.\n"
        "6) JSON만 출력."
    )
    resp = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": usr},
                {"type": "image_url", "image_url": {"url": b64}},
            ]},
        ],
    )
    return json.loads(resp.choices[0].message.content)

def ask_vision_with_conf(api_key: str, pil_img: Image.Image, model: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    b64 = pil_to_b64(pil_img)
    sys = "설명 없이 JSON만 출력. 각 키는 {'value':..., 'confidence':0~1} 형태."
    usr = (
        "표준키만 포함:\n" + list(FIELD_SCHEMA.keys()).__str__() +
        "\n각 항목은 {'value':<문자열>, 'confidence':<0~1 float>}.\n"
        "규칙: '제목' 우선, 결재선 무시, attachment_count는 숫자, 날짜/금액 포맷 맞추기."
    )
    resp = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": usr},
                {"type": "image_url", "image_url": {"url": b64}},
            ]},
        ],
    )
    return json.loads(resp.choices[0].message.content)

def gpt_extract_table(api_key: str, pil_img: Image.Image, model: str, use_confidence: bool) -> Dict[str, Any]:
    """전처리 + 2패스(모델 교차) + 값 정규화. 신뢰도 모드 시 confidence 포함 구조 허용."""
    img = preprocess_for_ocr(pil_img)

    # 1차
    if use_confidence:
        d1 = ask_vision_with_conf(api_key, img, model)
        d1v = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in d1.items()}
    else:
        d1 = ask_vision_values(api_key, img, model)
        d1v = d1

    # 2차(교차 모델)
    alt = "gpt-4o" if model == "gpt-4o-mini" else "gpt-4o-mini"
    if use_confidence:
        d2 = ask_vision_with_conf(api_key, img, alt)
        d2v = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in d2.items()}
    else:
        d2 = ask_vision_values(api_key, img, alt)
        d2v = d2

    # 키/값 정규화
    d1v = _normalize_values(_normalize_keys(d1v))
    d2v = _normalize_values(_normalize_keys(d2v))

    # 필드별 선택(패턴 만족 우선)
    merged = {}
    for k in FIELD_SCHEMA.keys():
        v1, v2 = d1v.get(k), d2v.get(k)
        if v1 == v2:
            merged[k] = v1
        else:
            pat = FIELD_SCHEMA[k].get("pattern")
            def ok(v): return bool(re.fullmatch(pat, str(v))) if pat and v is not None else (str(v).strip() != "")
            merged[k] = v1 if ok(v1) else (v2 if ok(v2) else (v1 or v2 or ""))

    # 방어: 제목/첨부 누락 시 1회 재질의
    if (not merged.get("제목")) or (merged.get("attachment_count", 0) == 0):
        if use_confidence:
            d3 = ask_vision_with_conf(api_key, img, model)
            d3v = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in d3.items()}
        else:
            d3 = ask_vision_values(api_key, img, model)
            d3v = d3
        d3v = _normalize_values(_normalize_keys(d3v))
        for k in ["제목", "attachment_count"]:
            if (not merged.get(k)) and d3v.get(k):
                merged[k] = d3v.get(k)

    if "제목" not in merged: merged["제목"] = ""
    if "attachment_count" not in merged: merged["attachment_count"] = 0
    return merged

# ================== 레퍼런스(빈칸 전용) ==================
def load_reference_stats_blank_only(pass_dir=PASS_DIR, fail_dir=FAIL_DIR, required_threshold=0.8):
    def _load_all(p):
        out=[]
        for f in glob.glob(os.path.join(p, "*.json")):
            try: out.append(json.load(open(f, "r", encoding="utf-8")))
            except: pass
        return out

    pass_docs = _load_all(pass_dir)
    fail_docs = _load_all(fail_dir)

    # PASS에서 채워진 비율
    filled_ratio = {}
    if pass_docs:
        keys = set().union(*[d.keys() for d in pass_docs])
        for k in keys:
            vals = [(str(d.get(k, "")).strip() != "") for d in pass_docs]
            if vals:
                filled_ratio[k] = sum(vals) / len(vals)

    # 사실상 필수(기본 0.8)
    inferred_required = {k for k, r in filled_ratio.items() if r >= required_threshold}

    # FAIL에서 자주 비는 항목(참고)
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
    }

def report_blanks_only(doc_json: dict, ref_stats: dict):
    """레퍼런스 기준 '사실상 필수' 필드만 검사 → 빈칸이면 리포트"""
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
    use_confidence = st.toggle("신뢰도(Confidence) 모드 사용", value=False, help="필드별 confidence(0~1)를 활용한 보수적 처리")
    st.markdown("---")
    llm_help_on = st.toggle("LLM 설명 생성(권고문/이유) 사용", value=False)
    model_text = st.selectbox("설명용 LLM (텍스트)", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=not llm_help_on)
    st.markdown("---")
    if st.button("레퍼런스 로드/갱신"):
        st.session_state["ref_blank"] = load_reference_stats_blank_only()
        rs = st.session_state["ref_blank"]
        st.success(f"로드 완료 ✅  PASS={rs['pass_count']}, FAIL={rs['fail_count']}, "
                   f"추론된 필수필드={len(rs['inferred_required'])}")
        if rs["fail_empty_rank"]:
            st.caption("FAIL에서 자주 비던 필드: " + ", ".join([k for k,_ in rs["fail_empty_rank"]]))

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
            st.session_state["doc_json"] = doc_json
            st.session_state["last_img_hash"] = img_hash
            st.success("문서 인식 완료 ✅")

        # === 버튼 3개 ===
        if "doc_json" in st.session_state:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            b1, b2, b3 = st.columns(3)

            with b1:
                if st.button("PASS 샘플로 저장"):
                    ensure_dirs()
                    path = os.path.join(PASS_DIR, f"pass_{ts}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state["doc_json"], f, ensure_ascii=False, indent=2)
                    st.success(f"PASS 샘플 저장 완료: {path}")

            with b2:
                if st.button("FAIL 샘플로 저장"):
                    ensure_dirs()
                    path = os.path.join(FAIL_DIR, f"fail_{ts}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state["doc_json"], f, ensure_ascii=False, indent=2)
                    st.success(f"FAIL 샘플 저장 완료: {path}")

            with b3:
                if st.button("이 문서 빈칸 확인"):
                    # 레퍼런스 자동 로드 시도
                    ref = st.session_state.get("ref_blank")
                    if not ref:
                        st.session_state["ref_blank"] = load_reference_stats_blank_only()
                        ref = st.session_state["ref_blank"]

                    if ref.get("pass_count", 0) + ref.get("fail_count", 0) == 0:
                        st.warning("레퍼런스 데이터가 없습니다. PASS/FAIL 샘플을 저장한 뒤, 사이드바에서 '레퍼런스 로드/갱신'을 눌러주세요.")
                    else:
                        blanks = report_blanks_only(st.session_state["doc_json"], ref)
                        if not blanks:
                            st.success("빈칸 없음 ✅ (레퍼런스 기준 필수필드 모두 채워짐)")
                        else:
                            st.error(f"빈칸 {len(blanks)}건 발견 ❌")
                            with st.expander("빈칸 상세 보기", expanded=True):
                                for it in blanks:
                                    st.write(f"- **항목명**: {it['항목명']}\n  - 문제점: {it['문제점']}\n  - 수정 예시: {it['수정 예시']}")
                            # (옵션) LLM 설명
                            if llm_help_on and api_key:
                                with st.spinner("LLM이 간단 가이드를 작성 중..."):
                                    advice = llm_explain_blanks(api_key, model_text, blanks)
                                if advice:
                                    st.markdown("**🔎 입력 가이드 (LLM 요약)**")
                                    st.write(advice)

# -------- 오른쪽: 레퍼런스 현황/설명 --------
with col2:
    st.subheader("② 레퍼런스 현황 / 필수 필드 설명")
    ref = st.session_state.get("ref_blank")
    if not ref:
        st.info("사이드바의 [레퍼런스 로드/갱신]을 눌러 주세요.")
    else:
        st.write(f"- PASS 샘플: **{ref['pass_count']}**개, FAIL 샘플: **{ref['fail_count']}**개")
        st.write(f"- 추론된 '사실상 필수' 필드 수: **{len(ref['inferred_required'])}**개")
        if ref["inferred_required"]:
            st.write("필수로 간주된 필드 예:", ", ".join(list(ref["inferred_required"])[:8]) + (" ..." if len(ref["inferred_required"])>8 else ""))

        if llm_help_on and api_key and ref.get("filled_ratio"):
            if st.button("필수로 보는 이유 한 줄 설명(LLM)"):
                with st.spinner("LLM이 요약 설명 작성 중..."):
                    msg = llm_explain_required(api_key, model_text, ref["filled_ratio"], topk=5)
                if msg:
                    st.markdown("**📝 왜 필수로 보나요? (LLM 요약)**")
                    st.write(msg)
