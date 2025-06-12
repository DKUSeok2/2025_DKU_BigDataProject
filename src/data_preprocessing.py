import pandas as pd
import numpy as np
import glob
import os
from typing import List, Dict

class RestaurantDataProcessor:
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.df_list = []
        self.restaurants = []
        
    def load_all_csv_files(self) -> List[pd.DataFrame]:
        """모든 CSV 파일 로드"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        print(f"발견된 CSV 파일: {len(csv_files)}개")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # '방문자 유형' 컬럼이 있으면 제거
                if '방문자 유형' in df.columns:
                    df = df.drop('방문자 유형', axis=1)
                    print(f"'{os.path.basename(csv_file)}'에서 '방문자 유형' 컬럼 제거")
                
                # 필수 컬럼 확인
                required_cols = ['식당명', '작성일', '방문횟수', '리뷰 내용', '리뷰 태그']
                if all(col in df.columns for col in required_cols):
                    self.df_list.append((df, csv_file))
                    print(f"로드 완료: {os.path.basename(csv_file)} ({len(df)}개 리뷰)")
                else:
                    print(f"필수 컬럼 누락으로 스킵: {os.path.basename(csv_file)}")
                    
            except Exception as e:
                print(f"파일 로드 실패: {os.path.basename(csv_file)} - {e}")
                
        print(f"총 {len(self.df_list)}개 파일 성공적으로 로드")
        return self.df_list
    
    def extract_restaurant_info(self, restaurant_name: str) -> Dict:
        """식당명에서 정보 추출"""
        # 기본값 설정
        location = "건대"
        menu_type = "한식"
        atmosphere = "캐주얼"
        price_range = "1-3만원대"
        
        name = restaurant_name.strip()
        
        # 위치 정보 추출
        if "건대" in name:
            location = "건대"
        elif "홍대" in name:
            location = "홍대"
        elif "성수" in name:
            location = "성수"
            
        # 메뉴 타입 추정
        if any(keyword in name for keyword in ["돼지", "고기", "갈비", "삼겹", "곱창", "규카츠"]):
            menu_type = "고기류"
            atmosphere = "회식, 단체모임"
            price_range = "2-4만원대"
        elif any(keyword in name for keyword in ["면", "국수", "라면", "우동"]):
            menu_type = "면요리"
            atmosphere = "간편식, 혼밥"
            price_range = "1-2만원대"
        elif any(keyword in name for keyword in ["카츠", "일식", "스시", "회"]):
            menu_type = "일식"
            atmosphere = "데이트, 모임"
            price_range = "2-5만원대"
        elif any(keyword in name for keyword in ["빙수", "디저트", "카페"]):
            menu_type = "디저트"
            atmosphere = "데이트, 휴식"
            price_range = "1-2만원대"
        elif any(keyword in name for keyword in ["포케", "타코", "웨스턴"]):
            menu_type = "퓨전"
            atmosphere = "캐주얼, 데이트"
            price_range = "1-3만원대"
            
        return {
            "location": location,
            "menu_type": menu_type, 
            "atmosphere": atmosphere,
            "price_range": price_range
        }
    
    def preprocess_data(self) -> List[Dict]:
        """모든 CSV 파일에서 데이터 전처리"""
        self.load_all_csv_files()
        restaurant_data = []
        
        for idx, (df, csv_file) in enumerate(self.df_list, 1):
            # 결측값 처리
            df = df.fillna('')
            
            # 식당명 추출 (첫 번째 행에서)
            restaurant_name = df['식당명'].iloc[0] if len(df) > 0 else os.path.basename(csv_file)
            
            # 식당명 정리 (개행 문자 제거 등)
            clean_name_parts = []
            for part in str(restaurant_name).split('\n'):
                part = part.strip()
                if part and not any(word in part for word in ['예약', '톡톡', '쿠폰', '육류', '고기요리']):
                    clean_name_parts.append(part)
            
            clean_name = clean_name_parts[0] if clean_name_parts else f"식당_{idx}"
            
            # 식당 정보 추출
            restaurant_info = self.extract_restaurant_info(clean_name)
            
            # 리뷰 데이터 처리
            all_reviews = " ".join(df['리뷰 내용'].astype(str))
            all_tags = " ".join(df['리뷰 태그'].astype(str))
            
            # 통계 계산
            avg_visits = df['방문횟수'].mean() if '방문횟수' in df.columns else 1.0
            total_reviews = len(df)
            
            # 검색용 텍스트 생성
            search_text = f"""
            식당명: {clean_name}
            위치: {restaurant_info['location']} 건대입구역 근처
            메뉴: {restaurant_info['menu_type']}
            분위기: {restaurant_info['atmosphere']}
            가격대: {restaurant_info['price_range']}
            리뷰요약: {all_reviews[:500]}
            태그: {all_tags[:200]}
            """
            
            # 평점 추정 (리뷰 내용 기반)
            positive_words = ['맛있', '좋', '추천', '최고', '훌륭', '만족']
            negative_words = ['별로', '실망', '아쉽', '그냥', '보통']
            
            positive_count = sum(all_reviews.count(word) for word in positive_words)
            negative_count = sum(all_reviews.count(word) for word in negative_words)
            
            if positive_count > negative_count * 2:
                rating = 4.5
            elif positive_count > negative_count:
                rating = 4.0
            else:
                rating = 3.5
            
            restaurant_info_dict = {
                'id': idx,
                'name': clean_name,
                'location': restaurant_info['location'],
                'menu_type': restaurant_info['menu_type'],
                'atmosphere': restaurant_info['atmosphere'],
                'price_range': restaurant_info['price_range'],
                'rating': rating,
                'review_count': total_reviews,
                'avg_visits': avg_visits,
                'search_text': search_text.strip(),
                'summary': f"{restaurant_info['location']}의 {restaurant_info['menu_type']} 전문점. {restaurant_info['atmosphere']} 장소로 인기."
            }
            
            restaurant_data.append(restaurant_info_dict)
            print(f"처리 완료: {clean_name} ({total_reviews}개 리뷰)")
        
        print(f"\n전체 전처리 완료: {len(restaurant_data)}개 식당 데이터")
        return restaurant_data
    
    def get_top_tags(self, n=10) -> List[str]:
        """가장 많이 언급된 태그들 반환"""
        all_tags = []
        for df, _ in self.df_list:
            for tags in df['리뷰 태그'].dropna():
                all_tags.extend([tag.strip() for tag in str(tags).split(',')])
                
        from collections import Counter
        tag_counts = Counter(all_tags)
        return [tag for tag, count in tag_counts.most_common(n)]

if __name__ == "__main__":
    # 테스트
    processor = RestaurantDataProcessor("data/")
    restaurants = processor.preprocess_data()
    
    print("\n=== 식당 목록 ===")
    for restaurant in restaurants:
        print(f"ID: {restaurant['id']}")
        print(f"이름: {restaurant['name']}")
        print(f"위치: {restaurant['location']}")
        print(f"메뉴: {restaurant['menu_type']}")
        print(f"리뷰 수: {restaurant['review_count']}")
        print("-" * 50) 