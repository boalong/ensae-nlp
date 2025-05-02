from datasets import load_dataset
import pandas as pd

from src.generate_responses import generate_responses
from src.assess_responses import assess_responses
from src.similarity import compute_similarities
from src.uncertainty_and_confidence import uncertainty_and_confidence_scores


NUM_EXAMPLES = 10
NUM_ANSWERS = 3
MODELS_LIST = [
        ('qwen2.5', 'Qwen/Qwen2.5-7B-Instruct')
    ]


###########################################
### DATA PRE-PROCESSING
###########################################
# Load the validation split of TriviaQA, rc.nocontext subset
dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
df = pd.DataFrame(dataset)
df = df.drop_duplicates('question_id')[['question_id', 'question', 'answer']]
df = pd.DataFrame(dataset).iloc[:NUM_EXAMPLES].reset_index(drop=True) # to save costs
df['answer'] = df['answer'].apply(lambda x: x['normalized_value'])


###########################################
### MAIN STEPS
###########################################

# 1. Generate m=NUM_ANSWERS responses per question
# generate_responses(df, NUM_ANSWERS, MODELS_LIST)

# 2. Assess responses with Mistral API
# assess_responses()

# 3. Compute similarities scores
# compute_similarities()

# 4. Compute uncertainty and confidence estimates
uncertainty_and_confidence_scores(NUM_ANSWERS)

# 5. Evaluatation metrics: AUROC and AUARC