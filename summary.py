import json
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# ChatMistralAI 초기화 (API 키와 모델 설정)
llm = ChatOpenAI(
    api_key="",
    model="gpt-4o",
    temperature=0.7
)

def clean_json_output(response_str: str) -> str:
    """
    Removes triple backticks and any markdown formatting from the response.
    """
    cleaned_response = response_str.replace("```", "")
    cleaned_response = re.sub(r'^\s*json\s*', '', cleaned_response, flags=re.IGNORECASE)
    return cleaned_response.strip()

##############################################
# 통화 기록을 요약하는 함수
##############################################
def summarize_call_log(call_log: str) -> dict:
    summary_prompt = PromptTemplate(
        input_variables=["call_log"],
        template="""
Based on the following call record, generate a JSON summary with the following keys:
1. "recipient": the party that received the call.
2. "purpose": the purpose of the call.
3. "result": indicate "success" if the purpose was achieved, or "failure" otherwise.
4. "failureReason": if the call failed, describe the reason extracted from the call record (otherwise, leave empty or null).
5. "nextSteps": what the user should do next (for example, call back or make a decision).
6. "additionalDetails": any additional information the user should be aware of.

Call record: {call_log}

Output only the JSON summary without any extra text or explanation.
        """
    )
    chain = LLMChain(llm=llm, prompt=summary_prompt, verbose=True)
    response_str = chain.run(call_log=call_log)
    response_str = clean_json_output(response_str)
    try:
        summary = json.loads(response_str)
    except json.JSONDecodeError:
        print("Failed to parse JSON response in summarize_call_log:", response_str)
        raise ValueError("Failed to parse JSON response in summarize_call_log")
    return summary

##############################################
# 통합 실행 예제
##############################################
def main():
    # 예시 통화 기록 (실제 기록 내용에 맞게 수정 가능)
    call_record = (
       """Hello Tony, this is Seo calling to confirm our meeting time scheduled for 7pm.

0:00
I'm sorry, I can't go to the apartment.

0:06
I understand, Tony. Could you let me know if there's a different time that works better for you, or if there's a reason for the change? I want to make sure we have the correct details.

0:11
1590 ms
Because my car is broken, so can we make another appointment next week?

0:23
I see, Tony. Thank you for letting me know. I will note that your car is currently unavailable and that you'd prefer to reschedule for next week. I will verify the new appointment time and follow up with you soon. Is there anything else I can assist you with today?

0:33
483 ms
Oh thank you Dan! I want you to let me know how faster when your appointment is available.

0:48
I'll make sure to check the earliest available appointment for next week and get back to you as soon as possible. If there's anything else you need, feel free to let me know.

0:59
556 ms
Oh thanks! It's okay.

1:08
Thank you, Tony. I'll follow up with you soon regarding the new appointment time. Have a great day!

1:12
Tool called: end_call


1:12
""")
    
    # 통화 기록 요약 생성
    summary = summarize_call_log(call_record)
    print("\n[Call Record Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
