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
            app = generate_AA.build_graph()
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


def make_train_val_test_data(args):
    pass


def run_document_writing_simulation(args):
    pass


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
    }

    func_list = args.func_list.split("/")

    print(func_list)

    for func in func_list:
        if func not in func_map:
            print('Func name: ', func)
            raise ValueError(f"Unknown function: {func}")

        func_map[func](args)
