# Uncertainty Quantification for Black-box LLMs

In this project, I reproduce the methodology of the article Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models. I used Qwen2.5 7B Instruct assessed by Mistral

## Structure of the project

Mettre un arbre et expliquer les principaux fichiers.

## Use this project

pip install -r requirements.txt

If you want to assess the answers of the LLM, you must create a .env file, place it at the root of the project, and put your Mistral credentials. Notice that it can be long due to rate limits of the free tier of the Mistral API.

It is recommended to run only the first step of the pipeline (generation of the responses) on a GPU. You can also change the first part if you want to use a black-box model.
To run the full pipeline on a GPU (not recommended because Mistral part is long and you will waste GPU time), you may want to modify the similarity computations because the way they are coded works only on a CPU.