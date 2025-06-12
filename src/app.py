import streamlit as st
import os
import sys
from typing import List, Dict, Tuple

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))

from vector_db import VectorDB
from llm_integration import SimpleRecommender, GemmaRecommender

class RestaurantRecommendationApp:
    def __init__(self):
        self.vector_db = None
        self.recommender = None
        self.use_llm = False
        
    @st.cache_resource
    def load_vector_db(_self):
        """벡터 DB 로드 (캐시됨)"""
        vector_db = VectorDB()
        
        # 벡터 DB 파일 확인
        index_path = "../models/faiss_index"
        if not os.path.exists(os.path.join(index_path, "faiss_index.index")):
            st.error("벡터 DB가 구축되지 않았습니다. src/vector_db.py를 먼저 실행해주세요.")
            return None
            
        vector_db.load_index(index_path)
        return vector_db
    
    @st.cache_resource
    def load_llm_recommender(_self):
        """LLM 추천기 로드 (캐시됨)"""
        try:
            return GemmaRecommender()
        except Exception as e:
            st.warning(f"LLM 로딩 실패: {e}")
            return None
    
    def run(self):
        # 페이지 설정
        st.set_page_config(
            page_title="🍽️ AI 맛집 추천 챗봇",
            page_icon="🍽️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 메인 제목
        st.title("🍽️ AI 기반 맛집 추천 챗봇")
        st.markdown("---")
        
        # 사이드바 설정
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # LLM 사용 여부
            st.markdown("#### 🤖 AI 추천 모드")
            st.info("Qwen2.5-1.5B-Instruct 모델을 사용한 고급 AI 추천을 제공합니다. 1.5B 파라미터로 가볍고 한국어 지원이 우수합니다!")
            
            self.use_llm = st.checkbox(
                "🚀 Qwen AI 추천 사용", 
                value=False,
                help="Qwen2.5-1.5B-Instruct 모델로 자연스러운 추천글 생성 (1.5B 파라미터, 한국어 특화)"
            )
            
            # 검색 개수 설정
            num_results = st.slider("추천 식당 개수", 1, 5, 3)
            
            st.markdown("---")
            st.markdown("### 📖 사용법")
            st.markdown("""
            1. 원하는 조건을 자연어로 입력하세요
            2. 예시:
               - "건대에서 고기집 추천해줘"
               - "데이트하기 좋은 파스타집"
               - "회식 장소로 좋은 곳"
            3. 검색 버튼을 클릭하세요
            """)
        
        # 벡터 DB 로드
        if self.vector_db is None:
            with st.spinner("벡터 DB 로딩 중..."):
                self.vector_db = self.load_vector_db()
                
        if self.vector_db is None:
            return
            
        # LLM 로드 (필요시)
        if self.use_llm and self.recommender is None:
            with st.spinner("AI 모델 로딩 중... (최초 실행시 시간이 걸릴 수 있습니다)"):
                self.recommender = self.load_llm_recommender()
                if self.recommender is None:
                    self.use_llm = False
                    st.warning("LLM 로딩에 실패하여 간단 모드로 전환됩니다.")
        
        # 메인 컨텐츠
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 검색 입력
            st.subheader("🔍 맛집 검색")
            user_query = st.text_input(
                "어떤 식당을 찾고 계신가요?",
                placeholder="예: 건대에서 데이트하기 좋은 고기집",
                key="search_query"
            )
            
            search_button = st.button("🔍 검색하기", type="primary")
            
        with col2:
            st.subheader("💡 추천 키워드")
            
            # 빠른 검색 버튼들
            if st.button("🥩 건대 고기집"):
                self.perform_search("건대 고기집 추천해줘", num_results)
                
            if st.button("🍝 데이트 파스타"):
                self.perform_search("데이트하기 좋은 파스타집", num_results)
                
            if st.button("🍻 회식 장소"):
                self.perform_search("회식하기 좋은 식당", num_results)
                
            if st.button("🍣 고급 일식"):
                self.perform_search("고급스러운 일식집", num_results)
        
        # 검색 실행
        if search_button and user_query:
            self.perform_search(user_query, num_results)
            
        # 샘플 데이터 표시
        with st.expander("📊 현재 데이터베이스 정보"):
            if self.vector_db and self.vector_db.restaurants:
                st.write(f"**총 {len(self.vector_db.restaurants)}개 식당 데이터**")
                
                for restaurant in self.vector_db.restaurants:
                    st.write(f"- **{restaurant['name']}** ({restaurant['location']}) - {restaurant['menu_type']}")
    
    def perform_search(self, query: str, num_results: int):
        """검색 수행 및 결과 표시"""
        
        with st.spinner("검색 중..."):
            # 벡터 검색
            search_results = self.vector_db.search(query, k=num_results)
            
        if not search_results:
            st.warning("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
            return
            
        st.markdown("---")
        st.subheader("🎯 검색 결과")
        
        # 추천 텍스트 생성
        if self.use_llm and self.recommender:
            with st.spinner("AI가 추천글을 작성 중..."):
                try:
                    recommendation_text = self.recommender.generate_recommendation_text(query, search_results)
                    st.markdown("### 🤖 AI 추천")
                    st.markdown(recommendation_text)
                except Exception as e:
                    st.error(f"AI 추천 생성 실패: {e}")
                    self.show_simple_results(search_results)
        else:
            # 간단한 결과 표시
            simple_recommender = SimpleRecommender()
            recommendation_text = simple_recommender.generate_recommendation_text(query, search_results)
            st.markdown(recommendation_text)
            
        # 상세 결과 표시
        st.markdown("---")
        st.subheader("📋 상세 정보")
        
        for i, (restaurant, score) in enumerate(search_results, 1):
            with st.expander(f"{i}. {restaurant['name']} (유사도: {score:.3f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**📍 위치:** {restaurant['location']}")
                    st.write(f"**🍽️ 메뉴:** {restaurant['menu_type']}")
                    st.write(f"**🎭 분위기:** {restaurant['atmosphere']}")
                    
                with col2:
                    st.write(f"**💰 가격대:** {restaurant['price_range']}")
                    st.write(f"**⭐ 평점:** {restaurant['rating']}/5.0")
                    st.write(f"**📝 리뷰:** {restaurant['review_count']}개")
                
                st.write(f"**💬 한줄평:** {restaurant['summary']}")

    def show_simple_results(self, search_results: List[Tuple[Dict, float]]):
        """간단한 결과 표시"""
        for i, (restaurant, score) in enumerate(search_results, 1):
            st.write(f"**{i}. {restaurant['name']}** (유사도: {score:.3f})")
            st.write(f"📍 {restaurant['location']} | 💰 {restaurant['price_range']} | ⭐ {restaurant['rating']}/5.0")
            st.write(f"🍽️ {restaurant['menu_type']} | 🎭 {restaurant['atmosphere']}")
            st.write(f"💬 {restaurant['summary']}")
            st.write("")

def main():
    app = RestaurantRecommendationApp()
    app.run()

if __name__ == "__main__":
    main() 