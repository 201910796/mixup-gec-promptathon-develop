# 🧪 MixUp_디벨롭 : Grammar Error Correction Promptathon

본 레포지토리는 Grammar Error Correction Promptathon에서 **Solar Pro API**를 활용해 **오류 유형 기반 RAG + 템플릿 프롬프트 설계** 방식으로 맞춤법 교정 성능을 향상시킨 실험을 재현하고 확장하기 위한 코드 및 가이드를 제공합니다.

---

## 📌 프로젝트 개요

* **목표**: Solar Pro API 기반의 LLM을 활용하여 오직 프롬프트 엔지니어링만으로 **정답 문장을 정확하게 복원**하는 맞춤법 교정 시스템 개발
* **접근 전략**:
  - 오류 유형(typo, spacing 등)을 분류하는 LLM 질의 → 해당 유형에 대한 예시 기반 RAG → 템플릿에 삽입
  - 예시 삽입형 단일턴 프롬프트 구성으로 교정 성능 개선
* **주요 실험 내용**:
  - 다양한 템플릿 톤(교정기/튜터/검수자) 비교
  - 단일턴 vs 멀티턴 성능 비교
  - 오류 유형별 RAG 예시 삽입이 성능에 미치는 영향 분석

---

## ⚙️ 환경 세팅 & 실행 방법
- 실험 환경: Python 3.9 Solar Pro

### 1. 레포지토리 클론

```bash
git clone https://github.com/your-org/grammar-correction-promptathon.git
cd grammar-correction-promptathon/code
```

### 2. 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. 실험 실행
```bash
python main.py
```

## 🧠 실험 전략 요약
- LLM에 오류 유형 질의 → prompts/rag_search.py 내 classify 함수 활용

- 오류 유형별 예시 검색 → Levenshtein 거리 기반 상위 K개 문장 검색

- 예시 삽입 템플릿 자동 구성 → templates.py에서 템플릿 동적 채움

- API 호출 및 결과 수집 → experiment.py에서 병렬 처리 및 재시도 구현

## 🚧 실험의 한계 및 향후 개선
### 한계
- 일부 문장에서 API 실패로 인해 공백 결과 발생

- 과교정 또는 교정 부족 사례 존재

- 대용량 데이터 처리 시 속도 및 비용 이슈

### 향후 개선 방향
- 429 Too Many Requests 회피 위한 비동기 큐 or 요청 제한 조절

- 교정 후보 비교용 후보군 생성 및 앙상블 전략

- API key 병렬 처리를 통한 실행 속도 개선

## 📂 폴더 구조
```bash
📁 code/
├── main.py              # 실험 실행 메인 스크립트
├── config.py            # 실험 설정 클래스
├── requirements.txt     # 패키지 의존성 목록
├── utils/
│   ├── experiment.py    # 실험 실행 로직 및 API 호출
│   └── metrics.py       # 평가 지표 (recall 등)
└── prompts/
    ├── templates.py     # 교정기 프롬프트 템플릿 정의
    └── rag_search.py    # 오류 유형 분류 및 예시 추출 RAG 함수
```

## ✍️ 팀원 정보
- 팀명: 디벨롭
- 팀장: 김지원
- 팀원: 권우주, 김형선

---
