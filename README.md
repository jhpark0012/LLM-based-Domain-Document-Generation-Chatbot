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

## File Structure
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
