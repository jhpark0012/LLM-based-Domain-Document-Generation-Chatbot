import time
from tqdm.auto import tqdm
import json
import ast
import re
import random
import yaml
import os
import io
import pickle
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

import my_parser
import generate_AA
import generate_CQ
import data_split
import simulation
import hf_dataset_io

# Basic Settings
BASE_DIR = Path("/workspace/experiment/Clarifying-Ambiguity-Project")
DATA_ROOT = BASE_DIR / "data/"

load_dotenv(BASE_DIR / ".env")

openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = "Clarifying-Ambiguity-Project"

SEED = 42
random.seed(SEED)


def load_prompts(key: str = "AA_prompt"):
    path_ = BASE_DIR / "scripts/prompt_templates.yaml"
    with open(path_, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        system_prompt = data[key]["system"]
        user_prompt = data[key]["user"]

        return system_prompt, user_prompt


def run_generate_AA(args):

    gen_sys, gen_user = load_prompts("Gen_AA_prompt")
    eval_sys, eval_user = load_prompts("Eval_AA_prompt")
    regen_sys, regen_user = load_prompts("Re_Gen_AA_prompt")

    case_name_list = args.case_name_list.split(" ")

    origin_Ans_path = DATA_ROOT / "Origin_Ans"
    for case_name in tqdm(case_name_list):
        print()
        print('==============================================')
        print(f'Case : {case_name}')
        print('==============================================')
        print()

        df = pd.read_excel(origin_Ans_path / f'{case_name}.xlsx')
        df.columns = df.columns.str.strip()

        df_aa = pd.DataFrame(columns=[
            '번호', '구분', '작성사항', '작성대상', '답변', 'lv1', 'lv2', 'lv3', 'iter'])  # 답변 결과 df

        # [사건명, 법원, 청구구분, 소가구분, 인격구분, 청구취지, 청구원인]
        domain_question_idx = [0, 1, 2, 3, 6, 33, 34]

        for idx in tqdm(domain_question_idx, desc='Generate AA'):
            row = df.iloc[idx]
            clear_answer = row['답변']
            question = ''
            for i, content in enumerate(['구분', '작성사항', '작성대상']):
                question += content + ': '
                question += str(row[content])
                if i < 2:
                    question += ', '

            print(f'\nQuestion {idx} : {question}')
            print(f'Answer : {clear_answer}')
            print()

            # 애매한 답변 생성 process
            app = generate_AA.build_aa_graph()
            initial_state = {
                "case_name": case_name,
                "question": question,
                "clear_answer": clear_answer,
                "max_iters": args.refine_num,
                "gpt_ver_gen": 'gpt-4o-mini',
                "gpt_ver_eval": 'gpt-5-mini',
                "judge_num": args.judge_num,
                "gen_temperature": args.gen_temperature,
                "eval_temperature": args.eval_temperature,
                "iter": 0,
                "gen_system_prompt": gen_sys,
                "gen_user_prompt": gen_user,
                "eval_system_prompt": eval_sys,
                "eval_user_prompt": eval_user,
                "regen_system_prompt": regen_sys,
                "regen_user_prompt": regen_user
            }

            final_state = initial_state
            for state in app.stream(initial_state, stream_mode="values"):
                if state["iter"] > final_state["iter"]:
                    print(f"State (Iter {state['iter']}):\n{state}\n\n\n")
                final_state = state

            aa_answer = final_state["aa_answer"]
            flag = final_state["flag"]

            if not flag:
                # Iteration을 다 돌아도 결국 Judge가 맞추지 못한 경우
                final_state["iter"] += 1

            # 애매한 답변 결과 저장
            result_ = row[['번호', '구분', '작성사항', '작성대상', '답변']].tolist()
            for lv, ans in aa_answer.items():
                result_.append(ans)
            result_.append(final_state["iter"])

            df_aa.loc[len(df_aa)] = result_

        os.makedirs(DATA_ROOT / "Ambig_Ans", exist_ok=True)
        df_aa.to_csv(DATA_ROOT / "Ambig_Ans" /
                     f'{case_name}_AA.csv', index=False)


def run_generate_CQ(args):

    gen_sys, gen_user = load_prompts("GEN_CQ_prompt")
    eval_sys, eval_user = load_prompts("Eval_CQ_prompt")
    regen_sys, regen_user = load_prompts("Re_Gen_CQ_prompt")

    case_name_list = args.case_name_list.split(" ")
    Ambig_Ans_path = DATA_ROOT / "Ambig_Ans"

    app = generate_CQ.build_cq_graph()

    for case_name in tqdm(case_name_list):

        start = time.time()
        print()
        print('==============================================')
        print(f'Case : {case_name}')
        print('==============================================')
        print()

        df = pd.read_csv(Ambig_Ans_path / f'{case_name}_AA.csv')
        df.columns = df.columns.str.strip()

        df_cq = pd.DataFrame(columns=[
            '번호', '구분', '작성사항', '작성대상', '답변',
            'ambig_ans', 'lv', 'cq', 'iter', 'final_report'
        ])

        len_df = len(df)
        for idx in tqdm(range(len_df)):
            clear_answer = df.loc[idx, '답변']

            # question 만들기 (작성대상 컬럼 공백/이름 혼재 대비해서 strip 컬럼 사용 권장)
            question = ''
            for i, content in enumerate(['구분', '작성사항', '작성대상']):
                question += content + ': '
                question += str(df.loc[idx, content])
                if i < 2:
                    question += ', '

            print(f'\nQuestion {idx}: {question}')
            print(f'Answer : {clear_answer}\n')

            for lv in ['lv1', 'lv2', 'lv3']:
                print(lv)
                aa_answer = df.loc[idx, lv]
                print(aa_answer)

                # 그래프 initial_state 구성
                initial_state = {
                    "case_name": case_name,
                    "question": question,
                    "clear_answer": clear_answer,
                    "aa_answer": aa_answer,

                    # loop/params
                    "cq_num": args.cq_num,
                    "judge_num": args.judge_num,
                    "max_iters": args.refine_num,
                    "threshold": args.cq_score_threshold,

                    # models/temps
                    "gpt_ver_gen": "gpt-4o-mini",
                    "gpt_ver_eval": "gpt-5-mini",
                    "gen_temperature": args.gen_temperature,
                    "eval_temperature": args.eval_temperature,

                    # prompts
                    "gen_system_prompt": gen_sys,
                    "gen_user_prompt": gen_user,
                    "eval_system_prompt": eval_sys,
                    "eval_user_prompt": eval_user,
                    "regen_system_prompt": regen_sys,
                    "regen_user_prompt": regen_user,

                    "iter": 0,
                }

                final_state = initial_state
                for state in app.stream(initial_state, stream_mode="values"):
                    if state["iter"] > final_state["iter"]:
                        print(f"State (Iter {state['iter']}):\n{state}\n\n\n")
                    final_state = state

                    final_rows = final_state.get("final_rows", [])

                for r in final_rows:
                    result_ = df.loc[idx, ['번호', '구분',
                                           '작성사항', '작성대상', '답변']].tolist()

                    result_.append(aa_answer)                 # ambig_ans
                    result_.append(lv)                        # lv
                    result_.append(r.get("final_cq", ""))     # cq
                    result_.append(r.get("iters_used", 0))    # iter
                    result_.append(r.get("final_report", {}))

                    df_cq.loc[len(df_cq)] = result_

        end = time.time()
        print(f'실행 시간 : {end-start}')

        os.makedirs(DATA_ROOT / "Golden_CQ", exist_ok=True)
        df_cq.to_csv(DATA_ROOT / "Golden_CQ" /
                     f'{case_name}_CQ.csv', index=False)

        break


def run_data_split(args):
    doc_template = pd.read_excel(DATA_ROOT / '답변데이터양식.xlsx')
    doc_template.columns = doc_template.columns.str.strip()

    path_dict = {'origin_ans_path': DATA_ROOT / 'Origin_Ans',
                 'golden_cq_path': DATA_ROOT / 'Golden_CQ'}

    train_case_name_list = ['사해행위취소', '손해배상_불법행위', '대여금_법인', '부당등기말소', '채권양도금', '용역대금',
                            '임대차보증금청구', '손해배상_미성년자원고', '건물인도청구_국가원고', '건물철거', '저당권말소', '채권양도금2']

    val_case_name_list = ['공유물분할', '임대차보증금반환_선정당사자']
    test_case_name_list = ['건물명도및손해배상', '공사대금청구의소']

    train, _ = data_split.make_dataset(
        doc_template, train_case_name_list, path_dict)
    val, _ = data_split.make_dataset(
        doc_template, val_case_name_list, path_dict)
    test, test_gold = data_split.make_dataset(
        doc_template, test_case_name_list, path_dict)

    final_data_csv_path = DATA_ROOT / 'Final_data/csv_files'
    final_data_jsonl_path = DATA_ROOT / 'Final_data/jsonl_files'
    os.makedirs(final_data_csv_path, exist_ok=True)
    os.makedirs(final_data_jsonl_path, exist_ok=True)

    train.to_csv(final_data_csv_path / 'train.csv', index=False)
    val.to_csv(final_data_csv_path / 'val.csv', index=False)
    test.to_csv(final_data_csv_path / 'test.csv', index=False)
    test_gold.to_csv(final_data_csv_path /
                     'test_gold.csv', index=False)

    train_dataset = data_split.make_dataset_json(train)
    val_dataset = data_split.make_dataset_json(val)
    test_dataset = data_split.make_dataset_json(test)

    with open(final_data_jsonl_path / "train_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(final_data_jsonl_path / "val_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in val_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(final_data_jsonl_path / "test_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in test_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_simulation(args):
    def run_one_row(
        app,  # graph.compile() 결과(=app)로 받는 걸 추천. graph라도 stream 지원하면 동일
        question: str, user_answer: str, golden_answer: str, df_history: pd.DataFrame, max_hops: int,
        gpt_ver_chatbot: str, gpt_ver_user_agent: str, chatbot_temperature: float, user_agent_temperature: float,
        chatbot_system_prompt: str, chatbot_user_prompt: str, user_agent_system_prompt: str, user_agent_user_prompt: str
    ):
        initial_state: simulation.SimState = {
            "question": question,
            "user_answer": user_answer,
            "user_answer_original": user_answer,
            "golden_answer": golden_answer,
            "df_history": df_history.to_string(index=False),

            "hops": 0,
            "max_hops": max_hops,
            "judgment": "",

            "selection_list": [],
            "excluded_options": [],
            "final_answer": None,

            "gpt_ver_chatbot": gpt_ver_chatbot,
            "gpt_ver_user_agent": gpt_ver_user_agent,
            "chatbot_temperature": chatbot_temperature,
            "user_agent_temperature": user_agent_temperature,

            "chatbot_system_prompt": chatbot_system_prompt,
            "chatbot_user_prompt": chatbot_user_prompt,

            "user_agent_system_prompt": user_agent_system_prompt,
            "user_agent_user_prompt": user_agent_user_prompt,
        }

        final_state = initial_state

        # hops 증가할 때만 출력
        for state in app.stream(initial_state, stream_mode="values"):
            if state["hops"] > final_state["hops"]:
                print(f"State (Hops {state['hops']}):\n{state}\n\n")
            final_state = state

        return (
            final_state["final_answer"],
            final_state["hops"],
            final_state.get("excluded_options", [])
        )

    if args.method == 'SFT':
        gpt_ver_chatbot = 'ft:gpt-4o-mini-2024-07-18:personal:markrv1:CXjmhXS7'
    else:
        gpt_ver_chatbot = 'gpt-4o-mini'

    chatbot_system_prompt, chatbot_user_prompt = load_prompts(
        f'{args.method}_prompt')
    user_agent_system_prompt, user_agent_user_prompt = load_prompts(
        'User_Agent_prompt')

    user_num_list = args.user_num_list.split("\n")

    save_dir = DATA_ROOT / f'Student_Data/simulation/{args.method}/'
    os.makedirs(save_dir, exist_ok=True)

    for user_num in user_num_list:
        user_num = int(user_num)
        df_user = pd.read_excel(
            DATA_ROOT / f'Student_Data/Ambig_Ans/{user_num}번.xlsx')
        df_user = df_user.iloc[3:, :].reset_index(drop=True)
        df_golden = pd.read_excel(
            DATA_ROOT / f'Student_Data/Golden_Ans/{user_num}번_정답.xlsx')

        df_res = df_user.copy()
        df_res['Golden Answer'] = df_golden['답변']

        final_answer_list = []
        hops_list = []

        for idx in tqdm(range(len(df_user)), desc="Simulation Start"):
            user_answer = df_user['답변'].iloc[idx]
            golden_answer = df_golden['답변'].iloc[idx]
            content_list = df_user[['구분', '작성사항', '작성대상 ']].iloc[idx].tolist()

            question = f'Q{idx}: ({content_list[2]}) 부분의 {content_list[1]} 작성 내용입니다. {content_list[0]}에 대해서 입력해주세요'

            print(f'\n[{idx}] Question : {question}')

            df_history = df_res[['구분', '작성사항',
                                '작성대상 ', 'Golden Answer']].iloc[:idx]

            app = simulation.build_simul_graph()
            final_answer, hops, _excluded = run_one_row(
                app=app,
                question=question,
                user_answer=user_answer,
                golden_answer=golden_answer,
                df_history=df_history,
                max_hops=args.refine_num,
                gpt_ver_chatbot=gpt_ver_chatbot,
                gpt_ver_user_agent='gpt-4o-mini',
                chatbot_temperature=args.gen_temperature,
                user_agent_temperature=args.eval_temperature,
                chatbot_system_prompt=chatbot_system_prompt,
                chatbot_user_prompt=chatbot_user_prompt,
                user_agent_system_prompt=user_agent_system_prompt,
                user_agent_user_prompt=user_agent_user_prompt,

            )

            final_answer_list.append(final_answer)
            hops_list.append(hops)

        df_res['Final Answer'] = final_answer_list
        df_res['Hops'] = hops_list
        df_res.to_csv(save_dir / f'df_{user_num}.csv', index=False)

        break


def run_hf_dataset_io(args):

    if args.hf_action == "push":
        result = hf_dataset_io.push_data(args.hf_path)
    elif args.hf_action == "pull":
        result = hf_dataset_io.pull_data(args.hf_path)
    else:
        raise ValueError(f"Invalid args.hf_action: {args.hf_action}")

    print(result)


if __name__ == "__main__":

    args = my_parser.parse_args()
    print(args)

    func_map = {
        "generate_AA": run_generate_AA,
        "generate_CQ": run_generate_CQ,
        "hf_dataset_io": run_hf_dataset_io,
        "data_split": run_data_split,
        "simulation": run_simulation
    }

    func_list = args.func_list.split("/")

    print(func_list)

    for func in func_list:
        if func not in func_map:
            print('Func name: ', func)
            raise ValueError(f"Unknown function: {func}")

        func_map[func](args)
