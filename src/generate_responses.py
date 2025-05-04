import gc
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def generate_responses(df, NUM_ANSWERS, MODELS_LIST):
    for model_name, model_id in MODELS_LIST:
        print(f"🔵 Loading model: {model_name}")
        
        # Clear CUDA cache before loading each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Report available memory
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"Free GPU memory: {free_mem:.2f} GiB")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                sliding_window=None
            )
            model.eval()

            outputs = []
            # Loop over prompts
            for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Answer these questions, return only the answer:\nQ: In Scotland a bothy/bothie is a?\nA: House\nQ: " + row['question'] + "\nA: (Just the answer)"}
                ]
                
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                tokenizer_output = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = tokenizer_output.input_ids.to(model.device)
                attention_mask = tokenizer_output.attention_mask.to(model.device)
                
                gen_config = {
                    "max_new_tokens": 512,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "do_sample": True,
                    "num_return_sequences": NUM_ANSWERS # Generate several responses at once
                }

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        **gen_config
                    )
                
                for i in range(generated_ids.shape[0]):
                    generated_text = tokenizer.decode(
                        generated_ids[i, input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    response = generated_text.strip()
                    
                    outputs.append({
                        'question_id': row['question_id'],
                        'question': row['question'],
                        'answer': row['answer'],
                        'model': model_name,
                        'completion_id': i,
                        'completion': response
                    })
                
                # Brief pause between prompts if processing multiple
                time.sleep(0.1)

                if (idx+1)%10 == 0:
                    # Save results
                    results_df = pd.DataFrame(outputs)
                    results_df.to_parquet(f'data/generated_responses/{model_name}_{str(idx//10).zfill(len(str(df.shape[0]))-1)}.parquet', index=False)
                    outputs = []
                    del results_df
            
            # Delete the model and tokenizer and clear cache when done
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            # Continue to the next model if one fails
            continue

    print("✅ All generations complete and saved!")
