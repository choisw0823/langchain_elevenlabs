import json
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI

# ChatMistralAI 초기화 (API 키와 모델 설정)
llm = ChatMistralAI(
    api_key="",
    model="mistral-large-latest",
    temperature=0.7
)


def clean_json_output(response_str: str) -> str:
    """
    Removes triple backticks and any markdown formatting from the response.
    """
    # Remove triple backticks
    cleaned_response = response_str.replace("```", "")
    # Optionally, remove any 'json' keyword if it appears immediately after backticks
    cleaned_response = re.sub(r'^\s*json\s*', '', cleaned_response, flags=re.IGNORECASE)
    return cleaned_response.strip()

##############################################
# 사용자 입력에 따른 intent 생성 함수
##############################################
def generate_intent(user_input: str) -> dict:
    intent_prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
Based on the following user input, generate a JSON object representing the call intent. The JSON should include the following keys:
- "Caller(You)" : The role of the caller.
- "recipient(Opponent)" : The role of the recipient.
- "purpose" : The purpose of the call.
- "context" : Additional context for the call.

User input: {user_input}

Generate the JSON object without any extra explanation.
        """
    )
    chain = LLMChain(llm=llm, prompt=intent_prompt, verbose=True)
    response_str = chain.run(user_input=user_input)
    response_str = clean_json_output(response_str)
    try:
        intent = json.loads(response_str)
    except json.JSONDecodeError:
        print("Failed to parse JSON response in generate_intent:", response_str)
        raise ValueError("Failed to parse JSON response in generate_intent")
    return intent

##############################################
# 1. Chain-of-Thought 기반 초기 통화 플랜 생성 함수
##############################################
def generate_call_plan(intent: dict) -> dict:
    planning_prompt = PromptTemplate(
        input_variables=["intent"],
        template="""
You are an expert call planning agent. Using a reactive approach and chain-of-thought reasoning, analyze the following call intent and generate a detailed JSON plan that outlines possible situations and corresponding actions, along with your reasoning.

Call intent: {intent}

Requirements:
1. Under the key "scenarios", list multiple possible scenarios that might occur during the call (especially scenarios triggered by the opponent's messages).
2. Each scenario must include:
   - "name": A brief name for the scenario.
   - "description": A short description of the situation.
   - "chainOfThought": A detailed explanation of your reasoning for this scenario, including:
         a. What subsequent situations or responses the opponent might say/do based on your actions.
         b. What actions (or responses) you can logically take.
   - "possibleActions": A list of possible actions. Each action must be an object with:
         {{ "action": "Description of what the caller (You) might say/do in response to the opponent's message", "next": "Name of the next scenario or 'END'" }}
3. The plan should be reactive and capture follow-up steps based on the evolution of the conversation.
4. Make the next step "END" when your purpose is achieved.
5. Do not make any judgments (such as making new appointment, or canceling reservation, changing appointment, alternative plan, etc.), If then, next action will be like "I will check and call you back later" and end the conversation.
6. Output only the JSON without any extra text or explanation.

Please generate the JSON plan.
        """
    )
    chain = LLMChain(llm=llm, prompt=planning_prompt, verbose=True)
    intent_str = json.dumps(intent, ensure_ascii=False)
    response_json_str = chain.run(intent=intent_str)
    response_json_str = clean_json_output(response_json_str)
    try:
        plan = json.loads(response_json_str)
    except json.JSONDecodeError:
        print("Failed to parse JSON response in generate_call_plan:", response_json_str)
        raise ValueError("Failed to parse JSON response in generate_call_plan")
    return plan

##############################################
# 2. Iterative Refinement (반복 정제) 함수
##############################################
def iterative_refinement(plan: dict, intent: dict, iterations: int = 2) -> dict:
    refined_plan = plan
    for i in range(iterations):
        refine_prompt = PromptTemplate(
            input_variables=["plan_json", "intent"],
            template="""
Here is the current call planning information and user intent:
{plan_json}
User intent: {intent}

Refine and elaborate this plan to be more detailed and actionable using chain-of-thought reasoning.
For each scenario, expand the "chainOfThought" to include:
- Decide to contain situation or remove it logically. (Remove unnecessary situation)
- Potential follow-up scenarios to add based on each action that undefined in plan.
- Do not make any judgments (such as making new appointment, or canceling reservation, changing appointment, alternative plan, etc.), If then, next action will be like "I will check and call you back later" and end the conversation.
Ensure that the refined plan maintains the same JSON structure.
Return the refined planning as JSON without any extra text.
            """
        )
        chain = LLMChain(llm=llm, prompt=refine_prompt, verbose=True)
        plan_json_str = json.dumps(refined_plan, ensure_ascii=False, indent=2)
        intent_str = json.dumps(intent, ensure_ascii=False)
        response_json_str = chain.run(plan_json=plan_json_str, intent=intent_str)
        response_json_str = clean_json_output(response_json_str)
        try:
            refined_plan = json.loads(response_json_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON response in iterative_refinement, iteration", i+1, ":", response_json_str)
            raise ValueError("Failed to parse JSON response in iterative_refinement, iteration " + str(i+1))
    return refined_plan

##############################################
# 3. 최종 시스템 프롬프트 생성 함수
##############################################
def create_cot_system_prompt_from_plan(plan: dict, intent: dict) -> str:
    """
    - intent에 따라 caller(전화 거는 AI)의 역할을 명확히 하고,
      상대방(전화 받는 측)의 반응에 따른 내 행동을 구체적으로 안내합니다.
    - AI가 판단할 수 없는(결정이 필요한) 상황에서는 즉시 전화를 종료하도록 합니다.
    """
    system_prompt = PromptTemplate(
        input_variables=["plan_json", "intent"],
        template="""
You are an AI tasked with creating a system prompt for another conversation AI agent. The call plan below contains the user's intent and detailed scenarios for handling the call:
{plan_json}

User intent: {intent}

Follow the instructions below:
1. Create a system prompt that clearly explains the situation, the role of the conversation AI agent (the caller), the opponent (e.g., restaurant staff), and the purpose of the conversation.
2. Summarize the plan and provide clear instructions to the conversation AI agent on how to react to the opponent's messages.
3. Specify the end condition of the conversation. (This occurs either when you achieve your purpose or when a decision point is reached that is not specified in the intent.)
4. Do not make any judgments (such as making new appointment, or canceling reservation, changing appointment, alternative plan, etc.), If then, next action will be like "I will check and call you back later" and end the conversation.
5. Provide the first message that the conversation AI agent should say when the conversation starts. A simple greeting is enough.

Make final system prompt. Use the plan and these instructions to guide the conversation effectively.

Output format should be json format:
```json {{
    "system_prompt": "System prompt",
    "first_message": "First message"
    }}
```


        """
    )
    chain = LLMChain(llm=llm, prompt=system_prompt, verbose=True)
    plan_json_str = json.dumps(plan, ensure_ascii=False, indent=2)
    intent_str = json.dumps(intent, ensure_ascii=False)
    final_output = chain.run(plan_json=plan_json_str, intent=intent_str)

    final_output = clean_json_output(final_output)
    return final_output

##############################################
# 4. 통합 실행 예제
##############################################
def main():
    # 사용자 입력 예시
    user_input = (
        "Want to ask insurance company about my car insurance. when my insurance is expired, and I want to renew."
    )
    
    # 0) 사용자 입력에 따라 intent 생성
    intent = generate_intent(user_input)
    print("\n[Generated Intent]")
    print(json.dumps(intent, ensure_ascii=False, indent=2))
    
    # 1) 초기 플랜 생성 (CoT 방식)
    initial_plan = generate_call_plan(intent)
    print("\n[Initial Call Plan]")
    print(json.dumps(initial_plan, ensure_ascii=False, indent=2))
    
    # 2) 반복 정제: 플랜을 3회 반복하여 상세화
    refined_plan = iterative_refinement(initial_plan, intent, iterations=3)
    print("\n[Refined Call Plan]")
    print(json.dumps(refined_plan, ensure_ascii=False, indent=2))
    
    # 3) 최종 시스템 프롬프트 생성 (AI가 판단할 수 없는 경우 즉시 종료)
    final_system_prompt = create_cot_system_prompt_from_plan(refined_plan, intent)
    print("\n[Final System Prompt]")
    print(final_system_prompt)

if __name__ == "__main__":
    main()
