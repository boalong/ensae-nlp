# Uncertainty Quantification for Black-box LLMs

This project is a reproduction of [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/abs/2305.19187) by Lin et al. It deals with uncertainty quantification for natural language generation using black-box LLMs, that is LLMs for which we only have access to the generated text.

## Table

[Overview](#overview)

[Structure of the project](#structure-of-the-project)

[Use this project](#use-this-project)

## Overview

This project implements and evaluates confidence and uncertainty measures proposed in the original paper, adapting them to work with Qwen2.5-7B-Instruct, with assessment provided by the Mistral API. It also includes alternative similarity measures to challenge the performance of the original paper.

### Methodology

The implementation follows the paper's framework for uncertainty quantification:

1. For a given input, multiple response samples are generated from the LLM
2. Pairwise similarity scores between responses are calculated
3. Uncertainty estimates and confidence scores are computed from these similarities

### Similarity metrics
  - Jaccard similarity
  - Levenshtein-based similarity [**new**]
  - NLI-based similarity (using entailment and contradiction scores)
  - embeddings-based similarity [**new**]

### Implementation details

This project uses:
- **Base LLM**: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Assement model**: [Mistral Large 24.11](https://docs.mistral.ai/getting-started/models/models_overview)
- **Dataset**: [triviaqa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

The implementation requires only black-box access to LLMs (no need for token-level logits or internal representations).

### Results

Results are detailed in the report.

## Structure of the project

The tree of this project is:

```
├── data
│   ├── assessed_responses
│   │   ├── qwen2.5_000.parquet
            ...
│   │   └── qwen2.5_099.parquet
│   ├── generated_responses
│   │   ├── qwen2.5_000.parquet
            ...
│   │   └── qwen2.5_099.parquet
│   ├── pairs_with_similarity
│   │   ├── qwen2.5_000.parquet
            ...
│   │   └── qwen2.5_099.parquet
│   └── responses_with_confidence
│       ├── qwen2.5_000.parquet
            ...
│       └── qwen2.5_099.parquet
├── img
│   └── AUARC.png
├── LICENSE
├── log
│   └── perf.txt
├── main.py
├── README.md
├── report.pdf
├── requirements.txt
├── results
│   └── results.csv
└── src
    ├── assess_responses.py
    ├── evaluation_metrics.py
    ├── generate_responses.py
    ├── similarity.py
    └── uncertainty_and_confidence.py
```

The most important file is `main.py` from which you can run the experiments. In the folder `src`, you will find modules containing the functions called in `main.py`. In the folder `data`, you will find:
- in `generated_responses`, the responses generated by [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- in `assessed_responses`, the responses assessed by the 24.11 version of [Mistral Large](https://docs.mistral.ai/getting-started/models/models_overview)
- in `pairs_with_similarity`, the similarity between each pairs of responses to the same question, computed with all methods detailed in the paper
- in `responses_with_confidence`, the confidence scores for each response, computed with all methods detailed in the paper.

## Use this project

To install dependencies:
```bash
pip install -r requirements.txt
```

If you want to assess the answers of the LLM, you must create a .env file, place it at the root of the project, and fill it with your Mistral credentials:
```
MISTRAL_API_KEY=##paste your API key here##
```

Notice that it can be long to assess all the answers due to rate limits of the Mistral API.

It is recommended to run only the first step of the pipeline (generation of the responses) on a GPU. You can also change the first part if you want to use a black-box model.
To run the full pipeline on a GPU (not recommended especially if you use the Mistral API), you may want to modify the similarity computations because they currently do not support GPU.