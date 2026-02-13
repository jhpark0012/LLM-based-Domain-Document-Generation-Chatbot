import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def make_dataset(doc_template, case_name_list, path_dict):

    # 애매한 거에 대한거
    ambig_user_content_list, ambig_assistant_content_list, ambig_lv_list = [], [], []
    for case_name in case_name_list:
        df_cq = pd.read_csv(
            path_dict['golden_cq_path'] / f'{case_name}_CQ.csv')

        merged_df = pd.merge(
            doc_template,
            df_cq,
            on=["번호", "구분", "작성사항", "작성대상"],
            how="inner"
        )

        for i in range(len(merged_df)):
            user_content = '질문: ' + \
                merged_df['질문'].iloc[i] + '\n' + \
                '답변 : ' + merged_df['ambig_ans'].iloc[i]
            try:
                assistant_content = '현재 답변은 애매한 답변입니다. ' + \
                    merged_df['cq'].iloc[i].split(':')[1].lstrip()
            except:
                assistant_content = '현재 답변은 애매한 답변입니다. ' + \
                    merged_df['cq'].iloc[i]
            ambig_user_content_list.append(user_content)
            ambig_assistant_content_list.append(assistant_content)
            ambig_lv_list.append(int(merged_df['lv'].iloc[i][2:]))

    df_ambig_all = pd.DataFrame(
        list(zip(ambig_user_content_list, ambig_assistant_content_list, ambig_lv_list)),
        columns=['user', 'assistant', 'lv']
    )

    # 정답에 대한거
    golden_user_content_list, golden_assistant_content_list, golden_lv_list = [], [], []
    for case_name in case_name_list:
        df_ca = pd.read_excel(
            path_dict['origin_ans_path'] / f'{case_name}.xlsx')

        df_ca.columns = df_ca.columns.str.strip()

        merged_df = pd.merge(
            doc_template,
            df_ca,
            on=["번호", "구분", "작성사항", "작성대상"],
            how="inner"
        )

        for i in range(len(merged_df)):
            user_content = '질문: ' + \
                merged_df['질문'].iloc[i] + '\n' + \
                '답변 : ' + merged_df['답변'].iloc[i]
            assistant_content = '현재 답변은 명확한 답변입니다.'
            golden_user_content_list.append(user_content)
            golden_assistant_content_list.append(assistant_content)
            golden_lv_list.append(0)

    df_golden_all = pd.DataFrame(
        list(zip(golden_user_content_list,
             golden_assistant_content_list, golden_lv_list)),
        columns=['user', 'assistant', 'lv']
    )
    df_all = pd.concat([df_ambig_all, df_golden_all])

    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_all, df_golden_all


def make_dataset_json(df):

    system_prompt = '''당신은 민사소송 소장 작성 도우미입니다.
        당신의 임무는 사용자의 **애매한 답변(Ambiguous Answer)**을 탐지하여,
        만약, 사용자의 답변이 애매할 경우, 그 의도를 명확히 파악하고 **정확한 작성 결과(Clear Answer)**로 이끌어낼 수 있는  
        가장 구체적이고 자연스러운 Clarifying Question (CQ)을 생성하세요.
        '''

    dataset = []

    for i in range(len(df)):
        tmp = {'messages': []}
        system_ = {'role': 'system', 'content': system_prompt}
        user_ = {'role': 'user', 'content': df.iloc[i]['user']}
        assistant_ = {'role': 'assistant', 'content': df.iloc[i]['assistant']}

        tmp['messages'].append(system_)
        tmp['messages'].append(user_)
        tmp['messages'].append(assistant_)

        dataset.append(tmp)

    return dataset
