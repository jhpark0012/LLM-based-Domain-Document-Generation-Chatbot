from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import Annotated, TypedDict, List, Dict, Any
import json
import warnings
warnings.filterwarnings("ignore")


# ----------------
# state update helpers
# ----------------
def last_value(old, new):
    return new


# ----------------
# voting / filtering (너 코드 그대로 이식)
# ----------------
def hard_vote_filter(eval_dict, threshold=8):
    """
    eval_dict 구조:
      {
        "Relevance": [[s1, s2, ...], ["r1","r2",...]],
        "Knowledgeable": [[...],[...]],
        "Naturalness": [[...],[...]],
        "Engagingness": [[...],[...]]
      }

    반환:
      eval_report: 각 카테고리별로 threshold 미만인 점수/이유만 추출한 dict
      flag: 모든 카테고리에서 탈락 항목 길이 ≤ 1 이면 True, 아니면 False
    """
    categories = ["Relevance", "Knowledgeable", "Naturalness", "Engagingness"]

    eval_report = {}
    for cat in categories:
        scores, reasons = eval_dict.get(cat, [[], []])

        if scores is None:
            scores = []
        if reasons is None:
            reasons = []

        if len(reasons) < len(scores):
            reasons = reasons + [""] * (len(scores) - len(reasons))

        filtered_scores = []
        filtered_reasons = []
        for s, r in zip(scores, reasons):
            sv = 0 if (s is None) else s
            if sv < threshold:
                filtered_scores.append(sv)
                filtered_reasons.append(r if r is not None else "")

        eval_report[cat] = [filtered_scores, filtered_reasons]

    flag = all(len(eval_report[cat][0]) <= 1 for cat in categories)
    return eval_report, flag


# ----------------
# State
# ----------------
class CQState(TypedDict):
    # inputs
    case_name: str
    question: str
    clear_answer: str
    aa_answer: str

    # configs
    cq_num: int
    judge_num: int
    max_iters: int
    threshold: int

    gpt_ver_gen: str
    gpt_ver_eval: str
    gen_temperature: float
    eval_temperature: float

    # prompts
    gen_system_prompt: str
    gen_user_prompt: str
    eval_system_prompt: str
    eval_user_prompt: str
    regen_system_prompt: str
    regen_user_prompt: str

    # runtime
    cq_list: Annotated[List[str], last_value]          # 최초 n개 CQ
    active_cq_idx: Annotated[int, last_value]         # 현재 다루는 CQ 인덱스
    model_cq: Annotated[str, last_value]              # 현재 CQ(재생성으로 계속 업데이트)
    iter: Annotated[int, last_value]                  # 현재 CQ의 iteration
    eval_report: Annotated[Dict[str, Any], last_value]
    flag: Annotated[bool, last_value]

    # outputs (선택) - 최종 결과 누적
    final_rows: Annotated[List[Dict[str, Any]], last_value]


# ----------------
# Nodes
# ----------------
def node_generate_cq_list(state: CQState) -> Dict:
    """
    cq_num 개 CQ 생성해서 cq_list에 저장
    """
    parser = JsonOutputParser()
    system = SystemMessagePromptTemplate.from_template(
        state["gen_system_prompt"])
    human = HumanMessagePromptTemplate.from_template(state["gen_user_prompt"])
    prompt = ChatPromptTemplate.from_messages([system, human])

    llm = ChatOpenAI(
        model=state["gpt_ver_gen"],
        temperature=state["gen_temperature"],
        model_kwargs={"response_format": {"type": "json_object"}},
        n=state["cq_num"],
    )

    messages = prompt.format_messages(
        case_name=state["case_name"],
        question=state["question"],
        clear_answer=state["clear_answer"],
        aa_answer=state["aa_answer"],
    )

    res = llm.generate([messages])
    gens = res.generations[0]

    cq_list: List[str] = []
    for gen in gens:
        raw = gen.message.content.strip()
        try:
            d = parser.parse(raw)
        except Exception:
            d = json.loads(raw)
        cq_list.append(d.get("CQ", ""))

    return {
        "cq_list": cq_list,
        "active_cq_idx": 0,
        "model_cq": cq_list[0] if len(cq_list) > 0 else "",
        "iter": 0,
        "final_rows": [],
        "flag": False,
        "eval_report": {},
    }


def node_eval_cq(state: CQState) -> Dict:
    """
    model_cq 평가(judge_num개) -> hard_vote_filter 적용
    """
    parser = JsonOutputParser()
    system = SystemMessagePromptTemplate.from_template(
        state["eval_system_prompt"])
    human = HumanMessagePromptTemplate.from_template(state["eval_user_prompt"])
    prompt = ChatPromptTemplate.from_messages([system, human])

    llm = ChatOpenAI(
        model=state["gpt_ver_eval"],
        temperature=state["eval_temperature"],
        model_kwargs={"response_format": {"type": "json_object"}},
        n=state["judge_num"],
    )

    messages = prompt.format_messages(
        case_name=state["case_name"],
        question=state["question"],
        clear_answer=state["clear_answer"],
        aa_answer=state["aa_answer"],
        model_cq=state["model_cq"],
    )

    res = llm.generate([messages])
    gens = res.generations[0]

    categories = ["Relevance", "Knowledgeable", "Naturalness", "Engagingness"]
    eval_dict = {cat: [[], []] for cat in categories}

    for gen in gens:
        raw = gen.message.content
        try:
            eval_res = parser.parse(raw)
        except Exception:
            eval_res = json.loads(raw)

        for cat in categories:
            cat_info = eval_res.get(cat, {})
            eval_dict[cat][0].append(cat_info.get("score", None))
            eval_dict[cat][1].append(cat_info.get("explanation", ""))

    final_report, flag = hard_vote_filter(
        eval_dict, threshold=state["threshold"])
    return {"eval_report": final_report, "flag": flag}


def node_regen_cq(state: CQState) -> Dict:
    """
    eval_report 기반 CQ 재생성
    """
    parser = JsonOutputParser()
    system = SystemMessagePromptTemplate.from_template(
        state["regen_system_prompt"])
    human = HumanMessagePromptTemplate.from_template(
        state["regen_user_prompt"])
    prompt = ChatPromptTemplate.from_messages([system, human])

    llm = ChatOpenAI(
        model=state["gpt_ver_gen"],
        temperature=state["gen_temperature"],
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    chain = prompt | llm | parser
    d = chain.invoke(
        {
            "case_name": state["case_name"],
            "question": state["question"],
            "clear_answer": state["clear_answer"],
            "aa_answer": state["aa_answer"],
            "model_cq": state["model_cq"],
            "eval_report": state["eval_report"],
        }
    )

    # 보통 {"CQ": "..."} 기대
    new_cq = d.get("CQ", "") if isinstance(d, dict) else ""
    return {"model_cq": new_cq, "iter": state["iter"] + 1}


def node_finalize_one_cq(state: CQState) -> Dict:
    """
    현재 CQ(model_cq)에 대한 iter loop 종료 후 결과를 final_rows에 누적
    """
    rows = list(state.get("final_rows", []))
    rows.append(
        {
            "active_cq_idx": state["active_cq_idx"],
            "final_cq": state["model_cq"],
            "iters_used": state["iter"] + (0 if state["flag"] else 1),
            "passed": bool(state["flag"]),
            "final_report": state["eval_report"],
        }
    )
    return {"final_rows": rows}


def node_next_cq(state: CQState) -> Dict:
    """
    다음 CQ로 이동 (있으면 model_cq 갱신, iter 초기화)
    """
    nxt = state["active_cq_idx"] + 1
    cq_list = state.get("cq_list", [])
    if nxt < len(cq_list):
        return {
            "active_cq_idx": nxt,
            "model_cq": cq_list[nxt],
            "iter": 0,
            "flag": False,
            "eval_report": {},
        }
    return {"active_cq_idx": nxt}


# ----------------
# Routers
# ----------------
def route_after_eval(state: CQState):
    """
    evaluate 후:
      - 합격이면 finalize_one_cq
      - 불합격 & iter < max_iters -> regenerate
      - 불합격 & iter >= max_iters -> finalize_one_cq
    """
    if state["flag"]:
        return "finalize_one_cq"
    if state["iter"] >= state["max_iters"]:
        return "finalize_one_cq"
    return "regenerate"


def route_after_finalize(state: CQState):
    """
    finalize_one_cq 후:
      - 다음 CQ 있으면 next_cq
      - 없으면 END
    """
    cq_list = state.get("cq_list", [])
    nxt = state["active_cq_idx"] + 1
    if nxt < len(cq_list):
        return "next_cq"
    return END


# ----------------
# Build Graph
# ----------------
def build_cq_graph():
    workflow = StateGraph(CQState)

    workflow.add_node("generate_cq_list", node_generate_cq_list)
    workflow.add_node("evaluate", node_eval_cq)
    workflow.add_node("regenerate", node_regen_cq)
    workflow.add_node("finalize_one_cq", node_finalize_one_cq)
    workflow.add_node("next_cq", node_next_cq)

    workflow.set_entry_point("generate_cq_list")

    workflow.add_edge("generate_cq_list", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {"regenerate": "regenerate", "finalize_one_cq": "finalize_one_cq"},
    )

    workflow.add_edge("regenerate", "evaluate")

    workflow.add_conditional_edges(
        "finalize_one_cq",
        route_after_finalize,
        {"next_cq": "next_cq", END: END},
    )

    workflow.add_edge("next_cq", "evaluate")

    return workflow.compile()
