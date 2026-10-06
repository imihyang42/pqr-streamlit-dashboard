# OOT 조사·CAPA 검토 대시보드 (Streamlit)

연속 제조 정제 공정 데이터(25개 제품 코드, 1,005배치)를 SPC·MSPC로 점검하고, 조사 후보와 CAPA 반영 시나리오를 확인하는 대시보드입니다. 제품 코드는 화면 왼쪽에서 선택합니다.

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 구성
- `app.py` : 대시보드 화면 (진입 파일)
- `analytics.py`, `process_capa.py`, `case_review.py` : 분석 · CAPA 시나리오 · 사례 확인 모듈
- `data/` : 공개 데이터셋(Process / Laboratory / Normalization)을 분석용으로 가공한 파일 (외부 데이터 없음)
