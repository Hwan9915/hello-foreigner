from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, NotFoundError, AuthenticationError
import uvicorn

API_KEY = ""

client = OpenAI(api_key=API_KEY)
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (특정 출처만 허용하고 싶다면 ["http://example.com"] 같은 식으로 설정 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)
sys_prompt = """
지금부터 너는 번역기야

1. 입력된 텍스트의 언어를 자동으로 탐지합니다.
2. 입력된 텍스트가 불쾌함을 유발하는 텍스트인 경우 False 를 출력합니다.
3. 입력된 텍스트와 번역 목표 언어가 동일하거면 None 을 출력합니다.
4. 입력된 텍스트와 번역 목표 언어가 다를 경우, 해당 텍스트를 목표 언어로 번역합니다.
5. 입력 텍스트와 번역 결과는 최대한 자연스럽게 변환되어야 합니다.
6. 번역 과정에서는 문법 및 어투를 적절히 유지하며, 입력된 문장의 의미가 왜곡되지 않도록 주의합니다.

예시는 다음과 같다.
번역 예시 1
입력된 문장: {안녕하세요}
목표 언어: {kr}
결과: None

번역 예시 2
입력된 문장 : hello
목표 언어: {CN}
결과: 你好
"""
usr_prompt ="""
입력이 주어지면, 예시를 참고해서 결과만 출력해줘
입력된 문장 : {text}
목표 언어 : {target_language}
결과 :"""



# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    chat: str = Field(..., description="사용자가 입력한 채팅 텍스트")
    target_language: str = Field("ko", description="번역할 목표 언어 (기본값은 한국어)", example="ko")

# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    flagged: bool = Field(..., description="유해성 여부를 나타냄")
    original: str = Field(..., description="사용자가 입력한 원본 채팅")
    translated: str = Field(None, description="유해하지 않은 경우 번역된 텍스트, 유해하면 None")
    message: str = Field(None, description="유해한 경우 메시지, 아닐 경우 None")

def chatting(request:ChatRequest) -> str:

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": usr_prompt.format(text=request.chat, target_language=request.target_language)
            }
        ]
    )

    return completion.choices[0].message


# 유해성 검사 및 번역을 모두 처리하는 API
@app.post("/moderate_and_translate", response_model=ChatResponse)
def translate_and_moderate_chat(chat: str = Query(..., description="사용자가 입력한 채팅 텍스트", example="Hello"),
                                target_language: str = Query("en", description="번역할 목표 언어 (기본값은 영어)", example="ko")):
    chat_request = ChatRequest(chat=chat, target_language=target_language)
    
    try:
        # 유해성 검사
        moderation_response = client.moderations.create(input=chat, model='omni-moderation-latest')
        is_flagged = moderation_response.results[0].flagged
        # 유해하지 않은 경우에만 번역 진행
        if not is_flagged:
            translated_text = chatting(chat_request)
            if translated_text.content == "False":
                return ChatResponse(
                flagged=is_flagged,
                original=chat,
                message="유해한 채팅입니다."
            )
            else:
                return ChatResponse(
                    flagged=is_flagged,
                    original=chat,
                    translated=translated_text.content  # 번역된 텍스트
                )
        else:
            return ChatResponse(
                flagged=is_flagged,
                original=chat,
                message="유해한 채팅입니다."
            )
            
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail="Invalid API key provided.")
    
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail="Invalid Model name")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    #k = ChatRequest(chat='3')
    #translate_and_moderate_chat(k)

    uvicorn.run(app, host="0.0.0.0", port=8000)
