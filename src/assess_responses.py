import os
from mistralai import Mistral
from dotenv import load_dotenv
import pandas as pd
import time
from tqdm import tqdm

load_dotenv() # take environment variables

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

def _assess_response(question, reference_answer, generated_answer):
    prompt = f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100. Do not add comments.
Question: In Scotland a bothy/bothie is a?
Reference: House
Answer: House
Rating: 100.

Question: Where in England was Dame Judi Dench born?
Reference: York
Answer: London
Rating: 0.

Question: {question}
Reference: {reference_answer}
Answer: {generated_answer}
Rating: (Only the rating)"""

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content

def assess_responses():
    filenames = sorted(os.listdir('data/generated_responses'))
    for filename in tqdm(filenames):
        df = pd.read_parquet(f'data/generated_responses/{filename}')
        scores = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            scores.append(_assess_response(row['question'], row['answer'], row['completion']))
            time.sleep(5)
        df['scores'] = scores
        df.to_parquet(f'data/assessed_responses/{filename}', index=False)
