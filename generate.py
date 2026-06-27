import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from architecture import LLaMABase

def generate_story(prompt: str, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize structural metadata (must exactly match training dimensions)
    vocab_size = 50257
    dim = 768
    num_layers = 6
    num_heads = 12
    num_kv_heads = 4
    max_seq_len = 256
    
    # 2. Fetch the tokenizer to decode the integer arrays
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 3. Instantiate model architecture and map weights from the optimization run
    model = LLaMABase(vocab_size, dim, num_layers, num_heads, num_kv_heads, max_seq_len).to(device)
    
    checkpoint_path = "llama_stories_final.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Weights target missing: {checkpoint_path}. Wait for train.py to conclude.")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Freeze layers, drop training dropout vectors
    
    print(f"\n📡 Context Prompt: '{prompt}'")
    print("⏳ Processing Token Matrix Generation...")
    
    # Encode input string to raw token tensor
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Autoregressive generation loop
    with torch.no_grad(): # Disable autograd graph tracking to maximize throughput speed
        for _ in range(max_new_tokens):
            # Crop context window if sequence exceeds maximum sequence dimensions
            x_cond = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]
            
            # Forward pass to fetch logits from final distribution matrix
            logits, _ = model(x_cond)
            
            # Pluck out the final logit token slice [Batch, Seq, Vocab] -> Last Token Step
            logits = logits[:, -1, :] / temperature
            
            # Filter logits matrix using Top-K truncation
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Map logit weights to clear mathematical probability distribution space
            probs = F.softmax(logits, dim=-1)
            
            # Multinomially sample the next index from the probability spectrum
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next index back to active tensor context block
            x = torch.cat((x, next_token), dim=1)
            
            # Break early if the model predicts the standard End of Text (EOS) marker
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Decode the final raw integer array back to coherent text strings
    output_text = tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
    print("\n📝 Model Output Generation:")
    print("-" * 50)
    print(output_text)
    print("-" * 50)

if __name__ == "__main__":
    import os
    # Run an inference pass with a standard child narrative prompt structure
    generate_story(prompt="Once upon a time, a girl named Lily found a shiny blue key in her garden.")
