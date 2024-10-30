import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포토폴리오 주제가 뭔가요?",
    "모델은 어떤 걸 썼나요?",
    "프로젝트 인원은 어떻게 되나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장은 누구인가요?",
    "어떤 데이터를 이용했나요?",
    "프로젝트 하는데 어려움은 없었나요?"
]

answers = [
    "YOLO를 이용해 장애인이 편하게 휠체어를 운전할 수 있게 도와주는 시스템을 만드는 것입니다.",
    "YOLO 모델 8버전을 썼습니다.",
    "이동근,손주용,박보은,석승연,김연우 5명입니다.",
    "총 3주입니다. 기획, 구현, 발표준비 등으로 구성했습니다.",
    "손주용입니다.",
    "직접 만든 영상과 AI허브의 데이터를 이용했습니다.",
    "많은 데이터를 라벨링하며 전처리하는 작업이 힘들었습니다. 하지만 결과가 잘 나와서 뿌듯했습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("프로젝트 챗봇")
# 이미지 표시
st.image("thumb.png", caption="Welcome to the Project Chatbot", use_column_width=True)
st.write("프로젝트에 관한 질문을 입력해보세요. 예: 프로젝트의 주제는 무엇인가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")

