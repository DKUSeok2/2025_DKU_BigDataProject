import faiss
import numpy as np
import pickle
import os
import glob
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from data_preprocessing import RestaurantDataProcessor

class VectorDB:
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        벡터 DB 초기화
        ko-sroberta-multitask: 한국어 특화 임베딩 모델
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.restaurants = []
        self.embeddings = None
        
    def build_index(self, restaurant_data: List[Dict]):
        """식당 데이터로부터 FAISS 인덱스 구축"""
        self.restaurants = restaurant_data
        
        # 검색용 텍스트들 추출
        texts = [restaurant['search_text'] for restaurant in restaurant_data]
        
        print("텍스트 임베딩 생성 중...")
        # 임베딩 생성
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # FAISS 인덱스 생성 (코사인 유사도를 위해 정규화)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도)
        
        # 임베딩 정규화 (코사인 유사도 계산을 위해)
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # 인덱스에 임베딩 추가
        self.index.add(normalized_embeddings.astype(np.float32))
        
        print(f"벡터 DB 구축 완료: {len(restaurant_data)}개 식당")
        
    def search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """쿼리에 대해 유사한 식당 검색"""
        if self.index is None:
            raise ValueError("인덱스가 구축되지 않았습니다. build_index()를 먼저 실행하세요.")
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 검색 수행
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # 결과 구성
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1:  # 유효한 인덱스인 경우
                restaurant = self.restaurants[idx].copy()
                results.append((restaurant, float(score)))
        
        return results
    
    def save_index(self, save_dir: str):
        """벡터 DB를 파일로 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.index"))
        
        # 식당 데이터 저장
        with open(os.path.join(save_dir, "restaurants.pkl"), "wb") as f:
            pickle.dump(self.restaurants, f)
            
        # 임베딩 저장
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        
        print(f"벡터 DB 저장 완료: {save_dir}")
    
    def load_index(self, save_dir: str):
        """저장된 벡터 DB 로드"""
        # FAISS 인덱스 로드
        self.index = faiss.read_index(os.path.join(save_dir, "faiss_index.index"))
        
        # 식당 데이터 로드
        with open(os.path.join(save_dir, "restaurants.pkl"), "rb") as f:
            self.restaurants = pickle.load(f)
            
        # 임베딩 로드
        self.embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
        
        print(f"벡터 DB 로드 완료: {len(self.restaurants)}개 식당")

def build_restaurant_db():
    """식당 벡터 DB 구축 메인 함수"""
    print("=== 식당 추천 벡터 DB 구축 (다중 CSV 파일) ===")
    
    # 1. 데이터 전처리 (모든 CSV 파일)
    processor = RestaurantDataProcessor("data/")
    restaurant_data = processor.preprocess_data()
    
    if not restaurant_data:
        print("처리할 식당 데이터가 없습니다!")
        return
    
    # 2. 벡터 DB 구축
    vector_db = VectorDB()
    vector_db.build_index(restaurant_data)
    
    # 3. 저장
    vector_db.save_index("models/faiss_index")
    
    # 4. 테스트 검색
    print("\n=== 테스트 검색 ===")
    test_queries = [
        "건대 고기집 추천해줘",
        "데이트하기 좋은 일식집",
        "혼밥하기 좋은 면요리집",
        "디저트 카페 추천",
        "회식 장소로 좋은 곳"
    ]
    
    for query in test_queries:
        print(f"\n검색어: '{query}'")
        results = vector_db.search(query, k=3)
        for i, (restaurant, score) in enumerate(results, 1):
            print(f"{i}. {restaurant['name']} (유사도: {score:.3f})")
            print(f"   위치: {restaurant['location']}, 메뉴: {restaurant['menu_type']}, 리뷰: {restaurant['review_count']}개")

if __name__ == "__main__":
    build_restaurant_db() 