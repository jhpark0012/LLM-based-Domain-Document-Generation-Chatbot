import os
import json
from typing import TypedDict, List, Optional, Dict, Any

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langgraph.graph import StateGraph, END


def make_llm(model: str, temperature: float = 0.7):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

# ---------------------------
# Graph State
# ---------------------------


class SimState(TypedDict):
    # --------------------
    # input data
    # --------------------
    question: str
    user_answer: str
    user_answer_original: str
    golden_answer: str
    df_history: str

    # --------------------
    # control
    # --------------------
    hops: int
    max_hops: int
    judgment: str

    # --------------------
    # options
    # --------------------
    selection_list: List[str]
    excluded_options: List[str]

    # --------------------
    # result
    # --------------------
    final_answer: Optional[str]

    # --------------------
    # model configs
    # --------------------
    gpt_ver_chatbot: str
    gpt_ver_user_agent: str
    chatbot_temperature: float
    user_agent_temperature: float

    # --------------------
    # prompts
    # --------------------
    chatbot_system_prompt: str
    chatbot_user_prompt: str

    user_agent_system_prompt: str
    user_agent_user_prompt: str

# ---------------------------
# LLM Calls
# ---------------------------


def node_chatbot_judge(state: SimState) -> SimState:
    """
    - user_answer가 명확한지 판단
    - 애매하면 5개 선택지 생성 (excluded_options 반영)
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            state['chatbot_system_prompt']),
        HumanMessagePromptTemplate.from_template(state['chatbot_user_prompt']),
    ])

    # SFT 모델
    llm = make_llm(state['gpt_ver_chatbot'],
                   temperature=state["user_agent_temperature"])

    # selection_list_prev 역할: excluded_options를 넘겨서 "이건 빼고 새로 만들어"를 유도
    selection_list_prev = state["excluded_options"]

    messages = prompt.format_messages(
        question=state["question"],
        user_ans=state["user_answer"],
        selection_list_prev=selection_list_prev
    )

    res = llm.generate([messages]).generations[0][0].text
    result = json.loads(res)

    # {"judgment": "...", "selection_list": [...]}
    judgment_raw = result.get("judgment", "")
    selection_list = result.get("selection_list", [])

    if "애매한" in judgment_raw:
        judgment = "ambig"
    elif "명확한" in judgment_raw:
        judgment = "clear"
    else:
        judgment = "ambig"

    state["judgment"] = judgment
    state["selection_list"] = selection_list[:5]

    return state


def node_user_agent_select(state: SimState) -> SimState:
    """
    user agent가 selection_list 중 하나를 고르거나 '없음'을 고름.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            state['user_agent_system_prompt']),
        HumanMessagePromptTemplate.from_template(
            state['user_agent_user_prompt']),
    ])

    llm = make_llm(state['gpt_ver_user_agent'],
                   temperature=state["chatbot_temperature"])

    messages = prompt.format_messages(
        df_history=state["df_history"],
        question=state["question"],
        user_answer=state["user_answer"],
        golden_answer=state["golden_answer"],
        selection_list=state["selection_list"],
    )

    res = llm.generate([messages]).generations[0][0].text
    result = json.loads(res)

    # Agent_prompt가 {"선택지": "..."} 형태라고 가정
    chosen = result.get("선택지", "없음")

    state["user_answer"] = chosen
    return state


# ---------------------------
# Control / Routing
# ---------------------------
def route_after_chatbot(state: SimState) -> str:
    """
    chatbot 판단 이후 다음 노드 결정
    """
    if state["judgment"] == "clear":
        return "finalize_clear"
    return "user_select"


def route_after_user_select(state: SimState) -> str:
    """
    user agent 선택 이후 다음 노드 결정
    """
    # user agent가 보기를 골랐으면 종료(최종 답)
    if state["user_answer"] != "없음":
        return "finalize_selected"

    # 없음을 고른 경우: hop 증가 후 재시도 or fallback
    if state["hops"] >= state["max_hops"]:
        return "finalize_fallback"
    return "prepare_regen"


def prepare_regen(state: SimState) -> SimState:
    """
    - hops 증가
    - 이번에 제시했던 선택지를 excluded_options에 누적
    - user_answer는 여전히 '없음' 상태로 두고, chatbot이 새로운 보기를 만들도록 유도
    """
    state["hops"] += 1
    # 이번 selection_list를 제외 목록에 누적
    state["excluded_options"] = list(
        set(state["excluded_options"] + state["selection_list"]))
    # user_answer는 '없음' 그대로 두되, 다음 라운드에서 chatbot이 excluded_options를 보고 새로 생성
    return state


def finalize_clear(state: SimState) -> SimState:
    state["final_answer"] = state["user_answer"]
    return state


def finalize_selected(state: SimState) -> SimState:
    state["final_answer"] = state["user_answer"]
    return state


def finalize_fallback(state: SimState) -> SimState:
    # 3번 초과(혹은 max_hops 도달) 시 "맨 처음 답변" 선택
    state["final_answer"] = state["user_answer_original"]
    return state


# ---------------------------
# Build Graph
# ---------------------------
def build_simul_graph():
    g = StateGraph(SimState)

    g.add_node("chatbot", node_chatbot_judge)
    g.add_node("user_select", node_user_agent_select)
    g.add_node("prepare_regen", prepare_regen)

    g.add_node("finalize_clear", finalize_clear)
    g.add_node("finalize_selected", finalize_selected)
    g.add_node("finalize_fallback", finalize_fallback)

    g.set_entry_point("chatbot")

    g.add_conditional_edges(
        "chatbot",
        route_after_chatbot,
        {
            "finalize_clear": "finalize_clear",
            "user_select": "user_select",
        }
    )

    g.add_conditional_edges(
        "user_select",
        route_after_user_select,
        {
            "finalize_selected": "finalize_selected",
            "prepare_regen": "prepare_regen",
            "finalize_fallback": "finalize_fallback",
        }
    )

    g.add_edge("prepare_regen", "chatbot")

    g.add_edge("finalize_clear", END)
    g.add_edge("finalize_selected", END)
    g.add_edge("finalize_fallback", END)

    return g.compile()


# ---------------------------
# Runner for one row
# ---------------------------
