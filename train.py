import os
import time
import math
import torch
from architecture import LLaMABase
from dataloader import PrepackedDataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend

def train():
    print("🚀 Initializing Master Training Engine...")
    
    # Hyperparameters for the TinyStories 80M Parameter Speedrun
    vocab_size = 50257
    dim = 768
    num_layers = 6
    num_heads = 12
    num_kv_heads = 4
    max_seq_len = 256  # Kept compact to accelerate token processing speed
    
    # Batch strategy to saturate 16GB VRAM cleanly
    batch_size = 32          # Micro-batch size per step
    grad_accum_steps = 4     # Accumulate over 4 steps = Effective Batch Size of 128
    max_steps = 2000         # Total training iterations
    
    # Optimization configurations
    learning_rate = 6e-4
    warmup_steps = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 Target Accelerator Context: {device.upper()}")
    
    # 1. Initialize data streams
    data_dir = "./data"
    train_loader = PrepackedDataLoader(data_dir, "train", batch_size, max_seq_len)
    val_loader = PrepackedDataLoader(data_dir, "validation", batch_size, max_seq_len)
    
    # 2. Instantiate Model
    model = LLaMABase(vocab_size, dim, num_layers, num_heads, num_kv_heads, max_seq_len).to(device)
    
    # 3. Setup Optimizer (Decay weights on matrices only, skip norms/biases)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    # Detect optimal AMP configuration for AMD card
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"⚡ Autocast Execution Type: {amp_dtype}")
    
    # Learning rate schedule helper (Cosine Decay with Linear Warmup)
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        if step > max_steps:
            return learning_rate * 0.1
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        conf = 0.5 * (1.0 + math.cos(math.pi * progress))
        return learning_rate * 0.1 + conf * (learning_rate - learning_rate * 0.1)

    # 4. Core Training Loop
    model.train()
    print("🏋️ Launching Optimization Loop...")
    
    for step in range(max_steps):
        t0 = time.time()
        
        # Dynamically update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        # Process gradient accumulation steps
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.get_batch(device)
            
            # Enforce hardware backend choice safely inside the loop
            backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            with sdpa_kernel(backends):
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device == "cuda")):
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps
                    
            loss_accum += loss.detach().item()
            loss.backward()
            
        # Clip gradients to prevent stability explosions
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if device == "cuda":
            torch.cuda.synchronize() # Wait for GPU execution to resolve for accurate step timing
            
        t1 = time.time()
        dt = (t1 - t0) * 1000 # milliseconds per step
        tokens_per_sec = (batch_size * max_seq_len * grad_accum_steps) / (t1 - t0)
        
        # Print Telemetry every 10 steps
        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {loss_accum * grad_accum_steps:.4f} | LR: {lr:.2e} | GradNorm: {norm:.4f} | Speed: {dt:.2f}ms | Throughput: {tokens_per_sec:.0f} tok/s")

    # Save final artifact checkpoint
    torch.save(model.state_dict(), "llama_stories_final.pt")
    print("✅ Training complete. Checkpoint saved to llama_stories_final.pt")

if __name__ == "__main__":
    train()
