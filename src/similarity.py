import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re
import Levenshtein
import os
import time
import numpy as np

def compute_similarities(NUM_EXAMPLES, methods=['jaccard', 'levenshtein', 'nli', 'sbert']):
    exec_times = {method: 0 for method in methods}
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

    if 'levenshtein' in methods:
        def _compute_levenshtein_similarity(s1, s2):
            s1 = s1.lower()
            s2 = s2.lower()
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 1.0
            distance = Levenshtein.distance(s1, s2)
            return 1 - distance / max_len

    if 'nli' in methods:
        # Load model and tokenizer
        model_name = "microsoft/deberta-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        nli_model.eval()
        def _compute_nli_similarity(premises, hypotheses):
            labels = ['contradiction', 'neutral', 'entailment']
            inputs = tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            results = [
                {label: float(prob) for label, prob in zip(labels, prob_vector)}
                for prob_vector in probs
            ]
            return results

    if 'sbert' in methods:
        sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        def _compute_sbert_similarity(sentences1, sentences2):
            embeddings1 = sim_model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = sim_model.encode(sentences2, convert_to_tensor=True)
            embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
            embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

            similarities = ((embeddings1 * embeddings2).sum(dim=1) + 1) / 2  # shift and rescale to [0, 1]
            return similarities

    filenames = sorted(os.listdir('data/generated_responses'))[:2]
    for filename in tqdm(filenames):
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
            start = time.perf_counter()
            pairwise_df['jaccard'] = pairwise_df.apply(lambda x: _compute_jaccard_similarity(x['left_response'], x['right_response']), axis=1)
            stop = time.perf_counter()
            exec_times['jaccard'] += (stop - start)

        if 'levenshtein' in methods:
            start = time.perf_counter()
            pairwise_df['levenshtein'] = pairwise_df.apply(lambda x: _compute_levenshtein_similarity(x['left_response'], x['right_response']), axis=1)
            stop = time.perf_counter()
            exec_times['levenshtein'] += (stop - start)

        if 'nli' in methods:
            start = time.perf_counter()
            nli = _compute_nli_similarity(pairwise_df['left_response'].tolist(), pairwise_df['right_response'].tolist())
            nli = pd.DataFrame(nli)
            pairwise_df['contra'] = 1 - nli['contradiction']
            pairwise_df['entail'] = nli['entailment']
            stop = time.perf_counter()
            exec_times['nli'] += (stop - start)

        if 'sbert' in methods:
            start = time.perf_counter()
            pairwise_df['sbert'] = _compute_sbert_similarity(pairwise_df['left_response'], pairwise_df['right_response'])
            stop = time.perf_counter()
            exec_times['sbert'] += (stop - start)

        pairwise_df.to_parquet(f'data/pairs_with_similarity/{filename}', index=False)

    # Log performance of methods to a file
    log_file = "log/perf.txt"
    with open(log_file, "w") as f:
        for method in methods:
            f.write(f"{method}: {exec_times[method]/(NUM_EXAMPLES/10):.6f} seconds\n")
