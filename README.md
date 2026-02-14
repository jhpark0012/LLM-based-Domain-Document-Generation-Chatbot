# LLM-based Domain Document Generation Chatbot
한양대학교 첨단제조산학협력프로젝트

## Background
In modern society, legal and administrative documents are essential, yet difficult for non-experts to write. Many users struggle with complex requirements, receive repeated revision requests, and restart the process multiple times.

Although LLMs like ChatGPT can help, they have limitations. They often fail to handle ambiguous user responses properly, may generate inaccurate legal content, and lack structured clarification mechanisms.

In legal domains such as civil complaints, users frequently provide vague answers due to limited domain knowledge.

Therefore, we propose a chatbot that detects ambiguity in user responses and generates Clarifying Questions (CQ) to collect precise information.


## File Structure
```bash
.
├── scripts/
│   ├── prompt_templates.yaml      # Task-specific prompt templates
│   └── run_main.sh                # Run main pipeline with configurable hyperparameters
│
├── src/
│   ├── main.py                    # Entry point that orchestrates and runs each module
│   ├── data_split.py              # Train/Validation/Test split logic
│   ├── generate_AA.py             # Ambiguous Answer (AA) data generation
│   ├── generate_CQ.py             # Clarifying Question (CQ) data generation
│   ├── hf_dataset_io.py           # Push/Pull datasets to and from Hugging Face Hub
│   ├── my_parser.py               # Argument and configuration parser
│   └── simulation.py              # Document writing simulation pipeline
├── data/                          # Dataset files
└── README.md
```

## Problem Definition
We focus on building a chatbot that can handle ambiguous user responses in domain-specific document writing.

Our approach consists of three main steps:

1) Construct synthetic datasets for the task

    - Generate Ambiguous Answers (AA)

    - Generate corresponding Clarifying Questions (CQ)

2) Train and evaluate the model

    - Fine-tune the model using the synthetic dataset (SFT)

    - Compare performance against baseline methods

3) Conduct real document-writing simulation

    - Evaluate document completion speed

    - Measure accuracy and clarification quality

Through this process, we aim to develop a chatbot that efficiently clarifies ambiguous inputs and accurately completes complex domain documents.

## Method
1. Synthetic Data Construction (AA, CQ)

본 연구에서는 도메인 문서 작성 과정에서 발생하는 **사용자 응답의 애매함(Ambiguity)**을 모델링하기 위해 합성 데이터셋을 구축하였다.

실험 도메인으로는 민사소송 소장을 선정하였다.
소장은 다음과 같은 두 유형의 질문으로 구성된다.

일반 질문 (예: 주소, 전화번호 등)

도메인 질문 (예: 사건명, 청구 구분, 인격 구분 등)

특히 도메인 질문의 경우, 법률 지식이 부족한 사용자는 애매한 표현으로 응답할 가능성이 높다.

이를 모델링하기 위해 다음 데이터를 구축하였다.

(1) Golden Answer (GA)

각 도메인 질문에 대해 명확하고 정답에 해당하는 Golden Answer를 정의하였다.

예시

Q: 사건명을 작성해주세요

A: 채권양도금

(2) Ambiguous Answer (AA)

비전문가 사용자의 응답을 모사하기 위해 LLM을 활용하여 **애매한 답변(Ambiguous Answer)**을 생성하였다.

예시

Q: 사건명을 작성해주세요

A: 돈을 다른 사람에게 넘기는 그런 경우

데이터 품질을 확보하기 위해 Feedback-Loop 기반 생성 파이프라인을 설계하였다.

<img src = "Img/aa_feedback_loop.png">
    - 데이터 생성 LLM이 애매한 답변 생성
    - 평가 LLM이 품질 및 적절성 평가
    - 미흡한 경우 피드백을 반영하여 재생성

이 과정을 반복함으로써 통제된 품질의 애매한 응답 데이터를 구축하였다.
