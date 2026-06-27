import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def prepack_data():
    print("🚀 Initializing High-Throughput Tokenization Pipeline...")
    
    # Define workspace path mapping
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Initialize the standard tokenizer
    print("📡 Fetching Tokenizer Config...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 2. Load the TinyStories text corpus from Hugging Face
    print("📥 Streaming TinyStories Dataset...")
    dataset = load_dataset("roneneldan/TinyStories", num_proc=os.cpu_count())
    
    # 3. Define the batch processing logic
    def process_split(split_name, dataset_split):
        print(f"⚡ Processing '{split_name}' split...")
        
        # Total number of tokens counter
        all_tokens = []
        
        # Tokenize in large string chunks to saturate CPU workers
        for batch in tqdm(dataset_split.iter(batch_size=1024), desc=f"Tokenizing {split_name}"):
            text_batch = batch["text"]
            
            # Map strings to raw token integers
            tokenized_batch = tokenizer(text_batch, add_special_tokens=False)
            
            for tokens in tokenized_batch["input_ids"]:
                # Append standard EOS (End of Text) token to separate stories cleanly
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token_id)
                
        # Convert to flat numpy array using uint16 (max value 65535, fits GPT2 vocab scale perfectly)
        token_array = np.array(all_tokens, dtype=np.uint16)
        
        # Write flat binary sequence straight to disk
        output_path = os.path.join(data_dir, f"{split_name}.bin")
        with open(output_path, "wb") as f:
            f.write(token_array.tobytes())
            
        print(f"💾 Saved {len(token_array):,} tokens to {output_path}")

    # Process both splits sequentially
    process_split("train", dataset["train"])
    process_split("validation", dataset["validation"])
    print("✅ Pre-packing step completed successfully.")

if __name__ == "__main__":
    prepack_data()
