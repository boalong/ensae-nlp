import numpy as np
import os
import pandas as pd

def uncertainty_and_confidence_scores(NUM_ANSWERS, methods=['jaccard', 'levenshtein', 'contra', 'entail', 'sbert']):
    filenames = sorted(os.listdir('data/generated_responses'))
    for filename in filenames:
        pairwise_df = pd.read_parquet(f'data/pairs_with_similarity/{filename}')

        for method in methods:
            pairwise_df[f'{method}_U_EigV'] = None
            pairwise_df[f'{method}_U_Deg'] = None
            pairwise_df[f'{method}_C_Deg'] = None
            pairwise_df[f'{method}_U_Ecc'] = None
            pairwise_df[f'{method}_C_Ecc'] = None

        num_pairs = NUM_ANSWERS*(NUM_ANSWERS-1) // 2
        for start in range(0, len(pairwise_df), num_pairs):
            chunk = pairwise_df.iloc[start:start+NUM_ANSWERS]

            for method in methods:

                W = np.zeros((NUM_ANSWERS, NUM_ANSWERS))
                for i, row in chunk.iterrows():
                    W[row['left_id'], row['right_id']] = row[method]
                W = W + W.T + np.eye(NUM_ANSWERS)
                D = np.diag(np.sum(W, axis=1))
                D_invhalf = np.diag(np.sum(W, axis=1)**(-0.5))

                L = np.eye(NUM_ANSWERS) - D_invhalf @ W @ D_invhalf
                eigenvalues, eigenvectors = np.linalg.eig(L)

                U_EigV = np.sum(np.maximum(0, 1 - eigenvalues))

                U_Deg = np.trace(NUM_ANSWERS*np.eye(NUM_ANSWERS) - D) / NUM_ANSWERS**2
                C_Deg = np.diag(D) / NUM_ANSWERS

                threshold = 0.9
                mask = eigenvalues < threshold
                filtered_eigenvectors = eigenvectors[:, mask]
                v_mean = filtered_eigenvectors.mean(axis=0, keepdims=True)
                v_primes = filtered_eigenvectors - v_mean
                U_Ecc = np.linalg.norm(v_primes)
                C_Ecc = - np.sqrt(np.sum(v_primes**2, axis=1))

                chunk[f'{method}_U_EigV'] = U_EigV
                chunk[f'{method}_U_Deg'] = U_Deg
                chunk[f'{method}_C_Deg'] = C_Deg
                chunk[f'{method}_U_Ecc'] = U_Ecc
                chunk[f'{method}_C_Ecc'] = C_Ecc

                pairwise_df.update(chunk)

        pairwise_df.to_parquet(f'data/responses_with_confidence/{filename}', index=False)
