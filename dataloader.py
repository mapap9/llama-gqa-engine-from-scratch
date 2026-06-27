import os
import torch
import numpy as np

class PrepackedDataLoader:
    """Zero-copy memory-mapped data reader for high-speed training streams."""
    def __init__(self, data_dir: str, split: str, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        bin_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing binary file: {bin_path}. Run tokenize_data.py first.")
            
        # Memory-map the binary array without loading it into RAM
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        # Calculate maximum token chunks available
        self.num_tokens = len(self.data)
        print(f"📖 Loaded {split} split with {self.num_tokens:,} tokens.")

    def get_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample random starting offsets across the whole dataset size
        # Leave headroom for (seq_len + 1) tokens so we have target pairs
        high = self.num_tokens - (self.seq_len + 1)
        offsets = torch.randint(0, high, (self.batch_size,))
        
        # Extract slices from memory map
        x_list = [torch.from_numpy((self.data[i : i + self.seq_len]).astype(np.int64)) for i in offsets]
        y_list = [torch.from_numpy((self.data[i + 1 : i + 1 + self.seq_len]).astype(np.int64)) for i in offsets]
        
        # Stack into batch tensors
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        
        # Push to targeted accelerator device memory asynchronously
        if "cuda" in device:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
            
        return x, y
