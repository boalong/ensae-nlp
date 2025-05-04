import pandas as pd
from datasets import load_dataset

from src.generate_responses import generate_responses
# from src.assess_responses import assess_responses
from src.similarity import compute_similarities
from src.uncertainty_and_confidence import uncertainty_and_confidence_scores
from src.evaluation_metrics import compute_all_metrics


NUM_EXAMPLES = 1000
NUM_ANSWERS = 10
MODELS_LIST = [
        ('qwen2.5', 'Qwen/Qwen2.5-7B-Instruct')
    ]

###########################################
### MAIN STEPS ############################
###########################################

# 1. Generate m=NUM_ANSWERS responses per question
# Load the validation split of TriviaQA, rc.nocontext subset
# dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
# df = pd.DataFrame(dataset)
# df = df.drop_duplicates('question_id')[['question_id', 'question', 'answer']]
# df = df.iloc[:NUM_EXAMPLES].reset_index(drop=True) # to save costs
# df['answer'] = df['answer'].apply(lambda x: x['normalized_value'])
# Generate responses
# generate_responses(df, NUM_ANSWERS, MODELS_LIST)

# 2. Assess responses with Mistral API
# assess_responses()

# 3. Compute similarities scores
# compute_similarities(NUM_EXAMPLES)

# 4. Compute uncertainty and confidence estimates
uncertainty_and_confidence_scores(NUM_ANSWERS)

# 5. Evaluatation metrics: AUROC and AUARC
# compute_all_metrics()
