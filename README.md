# 📄 AI 결재 사전검토 (제목+첨부파일 인식 자동판단)

가이드라인/유의사항 PDF를 임베딩(Chroma)에 저장하고, 결재 문서 이미지를 LLM(Vision)으로 파싱해 **제목/첨부 개수**를 자동 인식하고 규정 위반을 점검하는 Streamlit 앱입니다.

## 빠른 시작 (로컬)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # (Windows PowerShell) $Env:OPENAI_API_KEY="sk-..."
streamlit run app.py
# => http://localhost:8501
```

## Docker
```bash
docker compose up --build -d
# => http://localhost:8501
```

## 배포 (Streamlit Community Cloud)
1. 이 저장소를 GitHub에 올립니다.
2. Streamlit Cloud에서 **New app** → 저장소/브랜치/경로 선택.
3. App settings → **Secrets**에 `OPENAI_API_KEY` 추가.
4. 배포 URL을 팀과 공유하세요.

## 폴더 구조
```
repo/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
└─ chroma_db/
   └─ .gitkeep
```

## 메모
- 임베딩은 `chroma_db/`에 저장됩니다. 컨테이너/서버 재시작 시에도 유지하려면 볼륨 마운트 유지.
- 스캔 PDF는 `pypdf`로 텍스트가 안 뽑힐 수 있습니다. 필요 시 OCR 모듈을 추가하세요.
