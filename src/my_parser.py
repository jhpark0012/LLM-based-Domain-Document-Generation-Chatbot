import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run.", allow_abbrev=False)  # 약어 자동 매칭 비활성화

    parser.add_argument("--func_list", type=str)
    parser.add_argument("--hf_action", type=str,
                        choices=["push", "pull"], default="push")
    parser.add_argument("--hf_path", type=str)

    parser.add_argument('--case_name_list', type=str, default='임대차보증금청구',
                        help='Input case name')
    parser.add_argument('--refine_num', type=int,  default=3,
                        help='Input Max Number of Refinement')
    parser.add_argument('--judge_num', type=int,  default=3,
                        help='Input Number of Judge model')
    parser.add_argument('--cq_num', type=int,  default=3,
                        help='Input Number of CQ')
    parser.add_argument('--cq_score_threshold', type=int,  default=8,
                        help='Input Min score of CQ Quality')
    parser.add_argument('--gen_temperature', type=float,  default=0.7,
                        help='Input Temperature of Generative Model (GPT-4o-mini)')
    parser.add_argument('--eval_temperature', type=float,  default=0,
                        help='Input Temperature of Evaluation Model (GPT-5-mini)')

    parser.add_argument('--user_num', type=int,  default=1,
                        help='Input user num')

    return parser.parse_known_args()[0]
