from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from collections import Counter
import json
import random
import warnings
warnings.filterwarnings('ignore')


def last_value(old, new):
    """가장 최신 값 반영하는 함수"""
    return new


class AAState(TypedDict):
    case_name: str
    question: str
    clear_answer: str

    aa_answer: Annotated[Dict, last_value]
    eval_report: Annotated[List[Dict], last_value]
    flag: Annotated[bool, last_value]

    iter: Annotated[int, last_value]  # 현재 iter 수
    max_iters: int  # 최대 iter 수
    judge_num: int

    gpt_ver_gen: str
    gpt_ver_eval: str
    gen_temperature: float
    eval_temperature: float

    # Prompts
    gen_system_prompt: str
    gen_user_prompt: str
    eval_system_prompt: str
    eval_user_prompt: str
    regen_system_prompt: str
    regen_user_prompt: str


def shuffle_texts(d):
    items = list(d.items())  # [("lv1","..."), ("lv2","..."), ("lv3","...")]
    random.shuffle(items)
    texts = [t for _, t in items]
    gold = [k for k, _ in items]  # 정답 라벨 순서

    res = dict()
    for idx, t in enumerate(texts):
        res[idx] = t

    return res, gold


def hard_vote_one(item_by_judges: List[Dict], tie_break_idx: int = 0) -> Dict:
    """
    item_by_judges: 동일한 Ambiguous_Answer에 대한 각 judge의 평가 결과 리스트
    tie_break_idx: 동률일 경우 사용할 judge 인덱스 (0 = 1번 모델)
    """
    votes = [d["Judged_lv"] for d in item_by_judges]
    counts = Counter(votes)
    top_lv, top_cnt = counts.most_common(1)[0]
    ties = [lv for lv, c in counts.items() if c == top_cnt]

    # --- 다수결 승리 ---
    if len(ties) == 1:
        winner_lv = top_lv
        # 다수결로 선택된 lv를 낸 judge들의 reason 합치기
        reasons = []
        for i, d in enumerate(item_by_judges, start=1):
            if d["Judged_lv"] == winner_lv:
                reasons.append(f"Judge {i}: {d['Judgement_Reason']}")
        final_reason = " | ".join(reasons)
        chosen_judges = [
            i+1 for i, d in enumerate(item_by_judges) if d["Judged_lv"] == winner_lv]
    # --- 동률 시 ---
    else:
        winner_lv = item_by_judges[tie_break_idx]["Judged_lv"]
        final_reason = f"Judge {tie_break_idx+1}: {item_by_judges[tie_break_idx]['Judgement_Reason']}"
        chosen_judges = [tie_break_idx+1]

    ans = item_by_judges[0]["Ambiguous_Answer"]
    intended = item_by_judges[0].get("Intended_lv")

    return {
        "Ambiguous_Answer": ans,
        "Intended_lv": intended,
        "Final_lv": winner_lv,
        "Chosen_Judges": chosen_judges,
        "Final_Reason": final_reason
    }


def hard_vote_batch(judges_results: List[List[Dict]], tie_break_idx: int = 0):
    """
    judges_results: 예를 들어 [judge1_report, judge2_report, judge3_report]
    각 report는 [{'Ambiguous_Answer': ..., 'Intended_lv': ..., 'Judged_lv': ..., 'Judgement_Reason': ...}, ...]
    """
    n = len(judges_results[0])
    assert all(len(j) == n for j in judges_results), " judge의 샘플 수가 같아야 합니다."

    final_reports = []
    for i in range(n):
        item_by_judges = [judges_results[j][i]
                          for j in range(len(judges_results))]
        final_reports.append(hard_vote_one(
            item_by_judges, tie_break_idx=tie_break_idx))

    # 모든 Intended_lv == Final_lv 인지 확인
    flag = all(item["Final_lv"] == item["Intended_lv"]
               for item in final_reports)

    return final_reports, flag


def node_generate(state: AAState) -> Dict:
    system_prompt = state["gen_system_prompt"]
    user_prompt = state["gen_user_prompt"]

    parser = JsonOutputParser()
    system = SystemMessagePromptTemplate.from_template(system_prompt)
    human = HumanMessagePromptTemplate.from_template(user_prompt)
    prompt = ChatPromptTemplate.from_messages([system, human])

    llm = ChatOpenAI(model=state["gpt_ver_gen"], temperature=state["gen_temperature"], model_kwargs={
                     "response_format": {"type": "json_object"}})

    chain = prompt | llm | parser
    result = chain.invoke({
        "case_name": state["case_name"],
        "question": state["question"],
        "clear_answer": state["clear_answer"]
    })

    return {"aa_answer": result, "iter": 0}


def node_evaluate(state: AAState) -> Dict:
    system_prompt = state["eval_system_prompt"]
    user_prompt = state["eval_user_prompt"]

    # 1. Output Parser
    parser = JsonOutputParser()

    # 2. 메시지 템플릿 구성
    system = SystemMessagePromptTemplate.from_template(system_prompt)
    human = HumanMessagePromptTemplate.from_template(user_prompt)
    prompt = ChatPromptTemplate.from_messages([system, human])

    # 3. LLM 설정
    llm = ChatOpenAI(
        model=state["gpt_ver_eval"],
        temperature=state["eval_temperature"],
        model_kwargs={"response_format": {"type": "json_object"}},
        n=state["judge_num"],
    )

    # 텍스트 섞기
    texts, gold = shuffle_texts(state["aa_answer"])

    # 3.1 dict -> List[BaseMessage]
    messages = prompt.format_messages(
        case_name=state["case_name"],
        question=state["question"],
        clear_answer=state["clear_answer"],
        aa_answer=texts,
    )

    # 3.2 n개 생성
    res = llm.generate([messages])  # batch size 1
    gens = res.generations[0]       # length == judge_num

    eval_report_all = []

    # 각 판정자 응답 파싱 루프
    for gen in gens:
        raw = gen.message.content
        try:
            eval_res = parser.parse(raw)   # JSON을 dict로
        except Exception:
            eval_res = json.loads(raw)

        # 예상 형식: {"results": [{"label": "...", "why": "..."}, ...]}
        results = eval_res.get("results", [])
        pred = [r.get("label", "") for r in results]
        pred_reason = [r.get("why", "") for r in results]

        # 리포트 구성
        eval_report = []
        for k in range(len(texts)):
            eval_report.append({
                "Ambiguous_Answer": texts[k],
                "Intended_lv": gold[k],
                "Judged_lv": pred[k],
                "Judgement_Reason": pred_reason[k],
            })

        order = {"lv1": 1, "lv2": 2, "lv3": 3}
        eval_report_sorted = sorted(
            eval_report,
            key=lambda x: order.get(x["Intended_lv"], 999)
        )

        eval_report_all.append(eval_report_sorted)

    # 하드 보팅
    final_reports, flag = hard_vote_batch(eval_report_all)

    print(f'<Iteration Num : {state["iter"] + 1}>')
    print(state["aa_answer"])
    print()
    if flag:
        print('Good')
    else:
        print(final_reports)

    return {"eval_report": final_reports, "flag": flag}


def node_regenerate(state: AAState) -> Dict:
    system_prompt = state["regen_system_prompt"]
    user_prompt = state["regen_user_prompt"]

    parser = JsonOutputParser()
    system = SystemMessagePromptTemplate.from_template(system_prompt)
    human = HumanMessagePromptTemplate.from_template(user_prompt)
    prompt = ChatPromptTemplate.from_messages([system, human])

    llm = ChatOpenAI(model=state["gpt_ver_gen"], temperature=state["gen_temperature"], model_kwargs={
                     "response_format": {"type": "json_object"}})

    chain = prompt | llm | parser
    result = chain.invoke({
        "case_name": state["case_name"],
        "question": state["question"],
        "clear_answer": state["clear_answer"],
        "eval_report": state["eval_report"]
    })

    return {"aa_answer": result, "iter": state["iter"] + 1}


def route_after_evaluate(state: AAState):
    if state["flag"]:
        return END
    if state["iter"] >= state["max_iters"]:
        return END
    return "regenerate"


def build_graph():
    workflow = StateGraph(AAState)

    workflow.add_node("generate", node_generate)
    workflow.add_node("evaluate", node_evaluate)
    workflow.add_node("regenerate", node_regenerate)

    workflow.set_entry_point("generate")

    workflow.add_edge("generate", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {END: END, "regenerate": "regenerate"}
    )

    workflow.add_edge("regenerate", "evaluate")

    return workflow.compile()
