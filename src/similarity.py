import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re
import Levenshtein
import os

def compute_similarities(methods='jaccard|levenshtein|nli|sbert'):
    filenames = sorted(os.listdir('data/generated_responses'))
    for filename in filenames:
        df = pd.read_parquet(f'data/generated_responses/{filename}')

        pairwise_results = []
        # Iterate over all pairs
        for i, left_row in df.iterrows():
            for j, right_row in df.loc[i+1:].iterrows(): # avoid out of range error
                if left_row['question_id'] == right_row['question_id']:
                    pairwise_results.append({
                        "question_id": left_row['question_id'],
                        "left_id": left_row['completion_id'],
                        "right_id": right_row['completion_id'],
                        "left_response": left_row['completion'],
                        "right_response": right_row['completion']
                    })
                else:
                    continue

        pairwise_df = pd.DataFrame(pairwise_results)

        if 'jaccard' in methods:
            def _compute_jaccard_similarity(s1, s2):
                s1 = s1.lower()
                s2 = s2.lower()
                s1 = re.sub(r'[^ a-z0-9]', '', s1)
                s2 = re.sub(r'[^ a-z0-9]', '', s2)
                set1 = set(s1.split())
                set2 = set(s2.split())
                if len(set1) == 0 and len(set2) == 0:
                    return 1.0
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                return len(intersection) / len(union)

            pairwise_df['jaccard'] = pairwise_df.apply(lambda x: _compute_jaccard_similarity(x['left_response'], x['right_response']), axis=1)

        if 'levenshtein' in methods:
            def _compute_levenshtein_similarity(s1, s2):
                s1 = s1.lower()
                s2 = s2.lower()
                max_len = max(len(s1), len(s2))
                if max_len == 0:
                    return 1.0
                distance = Levenshtein.distance(s1, s2)
                return 1 - distance / max_len

            pairwise_df['levenshtein'] = pairwise_df.apply(lambda x: _compute_levenshtein_similarity(x['left_response'], x['right_response']), axis=1)

        if 'nli' in methods:
            # Load model and tokenizer
            model_name = "microsoft/deberta-large-mnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            nli_model.eval()
            def _compute_nli_similarity(premise, hypothesis):
                inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze()
                labels = ['contradiction', 'neutral', 'entailment']
                result = {label: float(prob) for label, prob in zip(labels, probs)}
                return result

            nli = pairwise_df.apply(lambda x: _compute_nli_similarity(x['left_response'], x['right_response']), axis=1)
            nli = nli.apply(pd.Series)
            pairwise_df['contra'] = 1 - nli['contradiction']
            pairwise_df['entail'] = nli['entailment']

        if 'sbert' in methods:
            sim_model = SentenceTransformer("all-MiniLM-L6-v2")
            def _compute_sbert_similarity(sentence1, sentence2):
                embeddings1 = sim_model.encode(sentence1)
                embeddings2 = sim_model.encode(sentence2)
                similarity = (sim_model.similarity(embeddings1, embeddings2) + 1) / 2 # shift and rescale to [0, 1]
                return similarity

            # A modifier pour faire vectoriel ?
            pairwise_df['sbert'] = pairwise_df.apply(lambda x: _compute_sbert_similarity(x['left_response'], x['right_response']).item(), axis=1)

        pairwise_df.to_parquet(f'data/pairs_with_similarity/{filename}', index=False)
