# 🍽️ AI 기반 식당 추천 챗봇

건국대학교 주변 맛집 데이터를 기반으로 한 벡터 검색과 LLM을 활용한 지능형 식당 추천 시스템입니다.

## 📋 프로젝트 개요

- **목적**: 사용자의 자연어 입력을 기반으로 건국대 주변 맛집을 추천
- **기술 스택**: FAISS (벡터 DB), ko-sroberta-multitask, Qwen2.5-1.5B-Instruct, Streamlit
- **데이터**: 건국대 주변 15개 맛집의 750개 실제 리뷰 데이터

## 🏗️ 시스템 아키텍처

```
사용자 입력 → 한국어 임베딩 → FAISS 벡터 검색 → Qwen2.5 LLM 추천 → Streamlit UI
```

## 📁 프로젝트 구조

```
2025_DKU_Bigdata/
├── data/                      # 15개 맛집 CSV 파일 (750개 리뷰)
│   ├── 최원석의돼지한판_서해쭈꾸미_건대1호점_*.csv
│   ├── 고베규카츠_건대점_*.csv
│   ├── 타코벨_건대스타시티점_*.csv
│   └── ... (총 15개 식당)
├── src/
│   ├── data_preprocessing.py   # 다중 CSV 데이터 전처리
│   ├── vector_db.py           # FAISS 벡터 DB 구축 및 검색
│   ├── llm_integration.py     # Qwen2.5 LLM 통합
│   └── app.py                 # Streamlit 웹 애플리케이션
├── models/
│   └── faiss_index/          # 저장된 벡터 인덱스 (gitignore)
├── project_report.md         # 프로젝트 보고서
├── requirements.txt          # 패키지 의존성
└── README.md
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# Conda 환경 생성 (권장)
conda create -n bigdata python=3.10
conda activate bigdata

# 의존성 설치
pip install -r requirements.txt
```

### 2. 벡터 데이터베이스 구축

```bash
cd src
python vector_db.py
```

### 3. 애플리케이션 실행

```bash
cd src
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

### 4. 브라우저에서 접속

- `http://localhost:8501` 또는 `http://192.168.0.204:8501`

## 💡 사용법

### 자연어 검색
1. 검색창에 원하는 조건을 자연어로 입력
2. 예시:
   - "건대에서 고기집 추천해줘"
   - "데이트하기 좋은 일식집"
   - "회식 장소로 좋은 곳"
   - "혼밥하기 좋은 면요리집"

### AI 추천 모드
- 사이드바에서 "🤖 AI 추천 사용" 체크
- Qwen2.5 모델이 자연스러운 추천글 생성

### 빠른 검색
- 우측의 추천 키워드 버튼 클릭

## 🔧 기술 세부사항

### 벡터 데이터베이스 (FAISS)
- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (한국어 특화)
- **유사도**: 코사인 유사도 (IndexFlatIP)
- **데이터**: 61개 식당 벡터 (15개 원본 + 확장 데이터)

### LLM 통합
- **모델**: Qwen2.5-1.5B-Instruct
- **실행**: 로컬 GPU 실행 (CUDA 지원)
- **특징**: 1.5B 파라미터로 가벼우면서 한국어 지원 우수

### 데이터 처리
- **다중 CSV 처리**: 15개 맛집의 개별 CSV 파일 통합
- **자동 전처리**: 컬럼 구조 차이 자동 처리
- **메뉴 분류**: 고기류, 면요리, 일식, 디저트, 퓨전, 한식 자동 분류

## 📊 데이터 구조

```python
restaurant_data = {
    'id': 1,
    'name': '최원석의돼지한판&서해쭈꾸미 건대1호점',
    'location': '건대',
    'menu_type': '고기류',
    'atmosphere': '회식, 단체모임',
    'price_range': '2-4만원대',
    'rating': 4.5,
    'review_count': 50,
    'avg_visits': 2.3,
    'search_text': '식당명, 위치, 메뉴, 분위기, 리뷰 통합 텍스트',
    'summary': '건대의 고기류 전문점. 회식, 단체모임 장소로 인기.'
}
```

## 🎯 주요 기능

1. **의미 기반 검색**: 벡터 유사도를 통한 정확한 매칭
2. **자연어 이해**: 복잡한 조건도 자연어로 처리
3. **AI 추천**: Qwen2.5 기반 개인화된 추천 텍스트
4. **실시간 UI**: Streamlit 기반 반응형 인터페이스
5. **다양한 카테고리**: 고기, 면요리, 일식, 디저트, 퓨전, 한식

## 🔍 예시 검색 결과

**입력**: "건대에서 데이트하기 좋은 일식집"

**출력**:
- 고베규카츠 건대점 (유사도: 0.92)
- 사토규카츠 건대본점 (유사도: 0.88)
- 위치: 건대, 메뉴: 일식, 분위기: 데이트, 모임

## ⚙️ 성능 및 최적화

### 시스템 요구사항
- **GPU**: NVIDIA GPU (CUDA 지원) 권장
- **메모리**: 8GB RAM 이상
- **저장공간**: 5GB 이상 (모델 캐시 포함)

### 처리 성능
- **벡터 검색**: 실시간 (< 1초)
- **LLM 추론**: 3-5초 (GPU 기준)
- **데이터 로딩**: 초기 1회 (약 10초)

## 🐛 문제 해결

### 벡터 DB 구축 오류
```bash
# 데이터 디렉토리 확인
ls data/*.csv

# 벡터 DB 재구축
cd src && python vector_db.py
```

### LLM 로딩 실패
- GPU 메모리 부족: AI 추천 기능 비활성화
- CUDA 오류: CPU 모드로 전환

### Streamlit 실행 오류
```bash
# 포트 변경
streamlit run app.py --server.port 8502

# 네트워크 접속 허용
streamlit run app.py --server.address=0.0.0.0
```

## 📈 확장 계획

1. **더 많은 식당**: 서울 전체 지역으로 확장
2. **실시간 데이터**: 운영시간, 대기시간 연동
3. **개인화**: 사용자 선호도 학습 및 반영
4. **멀티모달**: 음식 이미지 기반 추천
5. **지도 연동**: 위치 기반 시각화

## 🎓 교육적 가치

### 빅데이터 기술 학습
- **벡터 데이터베이스**: FAISS 활용법
- **자연어 처리**: 한국어 임베딩 모델 적용
- **LLM 통합**: 로컬 LLM 서비스 구축
- **웹 개발**: Streamlit을 통한 빠른 프로토타이핑

### 실전 프로젝트 경험
- **데이터 수집**: 실제 리뷰 데이터 활용
- **전처리**: 다양한 형태의 CSV 파일 처리
- **시스템 통합**: 여러 AI 기술의 조합
- **사용자 인터페이스**: 직관적인 웹 서비스 구현

## 📞 개발 정보

**개발 기간**: 2025년 6월  
**개발자** : 정다훈, 오유석
**목적**: 단국대학교 빅데이터 과목 프로젝트  
**기술 스택**: Python, FAISS, Transformers, Streamlit, CUDA  
**데이터**: 건국대 주변 15개 맛집, 750개 실제 리뷰  
**GitHub**: https://github.com/Downy-newlearner/2025_DKU_BigDataProject 
