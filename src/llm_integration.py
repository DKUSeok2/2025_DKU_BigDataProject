from transformers import pipeline
import torch
from typing import List, Dict, Tuple

class GemmaRecommender:
    def __init__(self):
        """Qwen2.5-1.5B-Instruct 모델 초기화 (Pipeline 방식)"""
        print("🚀 Qwen2.5-1.5B-Instruct 모델 로딩 중... (최초 실행시 다운로드로 시간이 걸릴 수 있습니다)")
        try:
            # Qwen2.5-1.5B-Instruct Pipeline 방식 (안정적이고 한국어 지원 우수)
            self.pipe = pipeline(
                "text-generation", 
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            
            print("✅ Qwen2.5-1.5B-Instruct 모델 로딩 완료!")
            print("💡 Qwen2.5는 1.5B 파라미터로 가볍고 한국어 지원이 우수합니다")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("💡 해결방법:")
            print("   1. transformers 버전 확인: pip install transformers>=4.50.0")
            print("   2. Hugging Face 로그인: huggingface-cli login")
            print("   3. 라이센스 동의: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct")
            raise e
    
    def generate_recommendation_text(self, query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """Qwen AI를 사용한 자연스러운 추천 텍스트 생성"""
        if not search_results:
            return "검색 결과가 없습니다."
        
        try:
            # 검색 결과 정리
            restaurants_info = []
            for i, (restaurant, score) in enumerate(search_results[:3], 1):
                restaurants_info.append(f"""
{i}. **{restaurant['name']}**
   - 위치: {restaurant['location']}
   - 메뉴: {restaurant['menu_type']}
   - 분위기: {restaurant['atmosphere']}
   - 가격대: {restaurant['price_range']}
   - 평점: {restaurant['rating']}/5.0
   - 한줄평: {restaurant['summary']}
""")
            
            restaurants_text = "\n".join(restaurants_info)
            
            # Qwen2.5를 위한 메시지 구성
            messages = [
                {
                    "role": "system", 
                    "content": "당신은 친근하고 도움이 되는 한국 맛집 추천 전문가입니다."
                },
                {
                    "role": "user", 
                    "content": f"""사용자 요청: "{query}"

추천할 식당들:
{restaurants_text}

위 식당들을 사용자 요청에 맞게 매력적으로 추천해주세요. 각 식당의 특별한 점과 추천 이유를 포함해서 자연스럽게 한국어로 설명해주세요. 간결하고 친근하게 작성해주세요."""
                }
            ]
            
            # Pipeline을 사용한 텍스트 생성
            outputs = self.pipe(
                messages, 
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # 응답 추출
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                
                if isinstance(generated_text, list) and len(generated_text) > 2:
                    # 마지막 메시지 (assistant 응답) 추출
                    assistant_response = generated_text[-1]["content"].strip()
                    if assistant_response:
                        return f"🤖 **Qwen AI 추천**\n\n{assistant_response}"
                elif isinstance(generated_text, str):
                    # 문자열인 경우 원본 프롬프트 이후 부분 추출
                    prompt_end = generated_text.find("간결하고 친근하게 작성해주세요.")
                    if prompt_end != -1:
                        response = generated_text[prompt_end + len("간결하고 친근하게 작성해주세요."):].strip()
                        if response:
                            return f"🤖 **Qwen AI 추천**\n\n{response}"
            
            # 응답이 없으면 폴백
            return self._fallback_response(query, search_results)
            
        except Exception as e:
            print(f"AI 추천 생성 중 오류: {e}")
            return self._fallback_response(query, search_results)
    
    def _fallback_response(self, query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """AI 실패시 폴백 응답"""
        simple_recommender = SimpleRecommender()
        return simple_recommender.generate_recommendation_text(query, search_results)

# 호환성을 위한 별칭
LlamaRecommender = GemmaRecommender

# 간단한 테스트용 클래스 (LLM 없이)
class SimpleRecommender:
    def generate_recommendation_text(self, user_query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """간단한 템플릿 기반 추천"""
        if not search_results:
            return "죄송합니다. 조건에 맞는 식당을 찾지 못했습니다."
        
        response = f"🔍 **'{user_query}'** 검색 결과\n\n"
        
        for i, (restaurant, score) in enumerate(search_results, 1):
            response += f"""
**{i}. {restaurant['name']}** ⭐ {restaurant['rating']}/5.0
📍 {restaurant['location']} | 💰 {restaurant['price_range']}
🍽️ {restaurant['menu_type']} | 🎭 {restaurant['atmosphere']}
📝 {restaurant['summary']}
📊 매칭도: {score:.1%}

"""
        
        response += "💡 **더 자연스러운 AI 추천을 원하시면 사이드바에서 'Gemma AI 추천 사용'을 체크해보세요!**"
        return response 