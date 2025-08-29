import os
import torch
import random
import pickle
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from torch.utils.data import Dataset
import math
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_asteroid import AsteroidTTSInstruct
from transformers import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
import argparse

# Import peft related modules
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
MAX_CHANNELS = 8

class PackedLazySupervisedDataset(Dataset):
    def __init__(self, data_dir, channels: int, tokenizer, max_input_tokens=2048, max_output_tokens=15000):
        super(PackedLazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.channels = channels
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        pkls = [os.path.join(data_dir, each) for each in os.listdir(data_dir) if each.endswith('.pkl')]
        self.raw_samples = []
        for pkl_file in pkls:
            metas = np.load(pkl_file.replace(".pkl", "_metas.npy"))
            pointers = metas[0]

            with open(pkl_file, "rb") as f:
                for start_pointer in pointers:
                    f.seek(int(start_pointer))
                    self.raw_samples.append(pickle.load(f))

        random.shuffle(self.raw_samples)
        print(f"Loaded {len(self.raw_samples)} raw samples")

        self.data = self.pack_all_samples()

    def pack_all_samples(self):
        packed = []
        buffer_input, buffer_label = [], []
        total_input_tokens = 0
        total_label_tokens = 0
        pack_lengths = []

        for example in self.raw_samples:
            input_ids = np.array(example["input_ids"])[:, :self.channels]
            labels = np.array(example["labels"])[:, :self.channels]

            shifted_input, shifted_labels = self.shift_example(input_ids, labels)

            input_len = shifted_input.shape[0]
            label_len = np.sum(shifted_labels[:,0] != -100)

            if (total_input_tokens + input_len > self.max_input_tokens) or (total_label_tokens + label_len > self.max_output_tokens):
                if buffer_input:
                    packed_example = self.create_packed_example(buffer_input, buffer_label)
                    packed.append(packed_example)
                    pack_lengths.append(packed_example["input_ids"].shape[0])
                buffer_input, buffer_label = [], []
                total_input_tokens = 0
                total_label_tokens = 0

            buffer_input.append(shifted_input)
            buffer_label.append(shifted_labels)
            total_input_tokens += input_len
            total_label_tokens += label_len

        if buffer_input:
            packed_example = self.create_packed_example(buffer_input, buffer_label)
            packed.append(packed_example)
            pack_lengths.append(packed_example["input_ids"].shape[0])

        if pack_lengths:
            avg_pack_length = sum(pack_lengths) / len(pack_lengths)
        else:
            avg_pack_length = 0
        print(f"Packed into {len(packed)} samples, average pack length: {avg_pack_length:.2f}")
        return packed

    def shift_example(self, input_ids, labels):
        """
        Perform delay shifting on a single sample, returning shifted_input_ids and shifted_labels
        """
        seq_len = input_ids.shape[0]
        new_seq_len = seq_len + self.channels - 1

        shifted_input_ids = np.full((new_seq_len, self.channels), 1024, dtype=np.int32)
        shifted_input_ids[:, 0] = self.tokenizer.pad_token_id
        shifted_labels = np.full((new_seq_len, self.channels), -100, dtype=np.int32)

        for i in range(self.channels):
            shifted_input_ids[i : (seq_len + i), i] = input_ids[:, i]
            shifted_labels[i : (seq_len + i), i] = labels[:, i]

        return shifted_input_ids, shifted_labels

    def create_packed_example(self, input_list, label_list):
        input_ids = np.concatenate(input_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        new_seq_len = input_ids.shape[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": np.ones(new_seq_len)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, np.ndarray]:
        line = self.data[i]
        
        # Data validation
        if "input_ids" not in line or "labels" not in line:
            raise ValueError(f"Data format error: sample {i} missing 'input_ids' or 'labels' field")
        return line


from torch.utils.data import IterableDataset, get_worker_info

class StreamingPackedDataset(IterableDataset):
    """
    Streamingly read .pkl + _metas.npy, loading only the currently needed samples each time;
    Online packing: accumulate until reaching max_input_tokens / max_output_tokens, then yield a packed sample.
    Support file/pointer sharding under multi-process DataLoader to avoid duplication.
    """
    def __init__(
        self,
        data_dir: str,
        channels: int,
        tokenizer: PreTrainedTokenizer,
        max_input_tokens: int,
        max_output_tokens: int = 15000,
        shuffle_files: bool = True,
        shuffle_within_file: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.channels = channels
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.shuffle_files = shuffle_files
        self.shuffle_within_file = shuffle_within_file
        self.seed = seed

        pkls = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]
        if not pkls:
            raise ValueError(f"No .pkl found in {data_dir}")
        self.pkl_path = pkls[0] 

        metas_path = self.pkl_path.replace(".pkl", "_metas.npy")
        metas = np.load(metas_path)
        self.pointers = metas[0].astype(np.int64)

        if self.shuffle_within_file:
            rnd = random.Random(self.seed)
            ptrs = list(self.pointers)
            rnd.shuffle(ptrs)
            self.pointers = np.array(ptrs, dtype=np.int64)

        self.total_samples = len(self.pointers)
        self.avg_samples_per_pack = 26

    def _shift_example(self, input_ids: np.ndarray, labels: np.ndarray):
        """Consistent with the original implementation: multi-channel delay shifting."""
        seq_len = input_ids.shape[0]
        new_seq_len = seq_len + self.channels - 1

        shifted_input_ids = np.full((new_seq_len, self.channels), 1024, dtype=np.int32)
        shifted_input_ids[:, 0] = self.tokenizer.pad_token_id
        shifted_labels = np.full((new_seq_len, self.channels), -100, dtype=np.int32)

        for i in range(self.channels):
            shifted_input_ids[i:(seq_len + i), i] = input_ids[:, i]
            shifted_labels[i:(seq_len + i), i] = labels[:, i]
        return shifted_input_ids, shifted_labels


    def _iter_stream(self, pkl_path, pointers):
        """Stream samples from a single pkl in pointer order."""
        with open(pkl_path, "rb") as f:
            for start in pointers:
                f.seek(int(start))
                example = pickle.load(f)  # Only load the current sample
                # Expected keys: input_ids, labels; shape should be [T, C_all], downstream will truncate to first channels
                input_ids = np.array(example["input_ids"])[:, :self.channels]
                labels = np.array(example["labels"])[:, :self.channels]
                yield self._shift_example(input_ids, labels)

    def _pack_generator(self, sample_iter):
        """
        Online packing: continuously accumulate shifted samples, yield a packed sample when exceeding the threshold.
        """
        buffer_input, buffer_label = [], []
        total_in, total_out = 0, 0

        def flush():
            nonlocal buffer_input, buffer_label, total_in, total_out
            if not buffer_input:
                return None

            # Concatenate the current buffer
            input_ids = np.concatenate(buffer_input, axis=0)   # [T, C]
            labels    = np.concatenate(buffer_label, axis=0)   # [T, C]
            T, C = input_ids.shape

            # Fixed length and channel-differentiated padding, consistent with collator
            FIX = self.max_input_tokens
            PAD_FIRST  = self.tokenizer.pad_token_id   # Padding for the first channel
            PAD_OTHER  = 1024                          # Padding for other channels (keep consistent with collator's filler_token_id)
            IGN        = -100                          # Padding for labels

            if T < FIX:
                pad_len   = FIX - T

                # input_ids: default all channels to 1024, then set the first channel to pad_token_id
                input_pad = np.full((pad_len, C), PAD_OTHER, dtype=np.int32)
                input_pad[:, 0] = PAD_FIRST
                input_ids = np.concatenate([input_ids, input_pad], axis=0)

                # labels: all channels -100
                label_pad = np.full((pad_len, C), IGN, dtype=np.int32)
                labels    = np.concatenate([labels, label_pad], axis=0)

                # attention_mask: pad zeros on the right
                attention = np.concatenate(
                    [np.ones(T, dtype=np.int32), np.zeros(pad_len, dtype=np.int32)]
                )

            else:
                # Defensive truncation to FIX (consistent with collator's upper limit)
                input_ids = input_ids[:FIX]
                labels    = labels[:FIX]
                attention = np.ones(FIX, dtype=np.int32)

            out = {
                "input_ids":      input_ids.astype(np.int32),  # [FIX, C]
                "labels":         labels.astype(np.int32),     # [FIX, C]
                "attention_mask": attention,                   # [FIX]
            }

            # Clear counters
            buffer_input.clear()
            buffer_label.clear()
            total_in = total_out = 0
            return out

        for shifted_input, shifted_labels in sample_iter:
            in_len = shifted_input.shape[0]
            out_len = int(np.sum(shifted_labels[:, 0] != -100))

            # If adding the next sample would exceed the threshold, yield the current pack first
            if (total_in + in_len > self.max_input_tokens) or (total_out + out_len > self.max_output_tokens):
                packed = flush()
                if packed is not None:
                    yield packed

            buffer_input.append(shifted_input)
            buffer_label.append(shifted_labels)
            total_in += in_len
            total_out += out_len

        # Flush remaining samples at the end
        packed = flush()
        if packed is not None:
            yield packed

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            sliced_pointers = self.pointers
        else:
            sliced_pointers = self.pointers[worker.id::worker.num_workers]

        def samples():
            yield from self._iter_stream(self.pkl_path, sliced_pointers)

        yield from self._pack_generator(samples())

    def __len__(self):
        # Return the estimated total number of packs, for Trainer display
        return math.ceil(self.total_samples / self.avg_samples_per_pack)

@dataclass
class DataCollatorForSupervisedDataset:
    pad_token_id: int
    max_length: int
    filler_token_id: int = 1024

    def __call__(self, instances: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        channels = input_ids[0].shape[1]
        max_length = min(max(ids.shape[0] for ids in input_ids), self.max_length)
        padded_input_ids, padded_labels, padded_attns = [], [], []
        
        for ids, lbls, attn in zip(input_ids, labels, attention_masks):
            seq_len = ids.shape[0]
            if seq_len < max_length:
                pad_len = max_length - seq_len
                input_pad = np.full((pad_len, channels), self.filler_token_id)
                input_pad[:, 0] = self.pad_token_id
                padded_input_ids.append(np.concatenate([ids, input_pad]))
                label_pad = np.full((pad_len, channels), -100)
                padded_labels.append(np.concatenate([lbls, label_pad]))
                attn_pad = np.zeros(pad_len)
                padded_attns.append(np.concatenate([attn, attn_pad]))
            else:
                padded_input_ids.append(ids[:max_length])
                padded_labels.append(lbls[:max_length])
                padded_attns.append(attn[:max_length])

        input_ids = torch.tensor(np.stack(padded_input_ids), dtype=torch.long)
        labels = torch.tensor(np.stack(padded_labels), dtype=torch.long)
        attention_mask = torch.tensor(np.stack(padded_attns), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def train(model_path : str, train_data_dir : str, eval_data_dir : str, output_dir : str, training_config : Dict, device: str = "cuda", use_lora: bool = False, lora_cfg: Dict = None):
    print("Initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained('fnlp/MOSS-TTSD-v0.5')
    tokenizer.padding_side = "left"
    
    # Load model with CPU offload support
    model = AsteroidTTSInstruct.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    
    # Move model to device first before any operations
    # model.to(torch.device(device))
    
    model.set_weights([8,2,1,1,1,1,1,1])
    model.config.use_cache = False
    
    # Enable gradient checkpointing first (on base model)
    if training_config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        print("Gradient checkpointing enabled")
    
    # Configure LoRA parameters if using LoRA
    if use_lora:
        print("Configuring LoRA parameters...")
        
        # Default LoRA configuration
        default_lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'lora_dropout': 0.05,
            'bias': "none",
            'use_rslora': True
        }
        
        # Merge with user provided configuration
        if lora_cfg:
            default_lora_config.update(lora_cfg)
        
        print(f"Using LoRA configuration: {default_lora_config}")
        
        lora_config = LoraConfig(
            r=int(default_lora_config['r']),
            lora_alpha=int(default_lora_config['lora_alpha']),
            target_modules=default_lora_config['target_modules'],
            lora_dropout=float(default_lora_config['lora_dropout']),
            bias=default_lora_config['bias'],
            task_type=TaskType.CAUSAL_LM,
            use_rslora=bool(default_lora_config['use_rslora']),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA configuration completed")
        
        # Re-enable gradient checkpointing on PEFT model (to ensure compatibility)
        if training_config.get('gradient_checkpointing', True):
            # Call base model's method
            model.base_model.gradient_checkpointing_enable()
            print("Re-enabled gradient checkpointing on LoRA base model")
        
        # Ensure model is in training mode and verify trainable parameters
        model.train()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise ValueError("No trainable parameters! LoRA configuration might be problematic.")
        print(f"Number of trainable parameters: {trainable_params:,}")
    else:
        model.train()
    
    print("Initializing dataloader")
    train_dataset = StreamingPackedDataset(
        train_data_dir, MAX_CHANNELS, tokenizer,
        max_input_tokens=training_config.get('max_input_tokens', 4096),
        max_output_tokens=training_config.get('max_output_tokens', 15000),
        shuffle_files=True, shuffle_within_file=False, seed=training_config.get('seed', 42)
    )
    eval_dataset = StreamingPackedDataset(
        eval_data_dir, MAX_CHANNELS, tokenizer,
        max_input_tokens=training_config.get('max_input_tokens', 4096),
        max_output_tokens=training_config.get('max_output_tokens', 15000),
        shuffle_files=True, shuffle_within_file=False, seed=1234
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer.pad_token_id, 32768)
    
    # =========================
    # TrainingArguments Categories and Annotations
    # =========================

    training_args = TrainingArguments(
        # 1. Output and logging related
        output_dir=output_dir,  # Training output directory
        logging_dir=os.path.join(output_dir, "logs"),  # Tensorboard logging directory
        report_to="tensorboard",  # Logging report method

        # 2. Training batch and steps related
        per_device_train_batch_size=int(training_config.get('per_device_train_batch_size')),  # Per-device training batch size
        per_device_eval_batch_size=int(training_config.get('per_device_eval_batch_size', training_config.get('per_device_train_batch_size'))),  # Per-device evaluation batch size
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps')),  # Gradient accumulation steps
        num_train_epochs=int(training_config.get('num_train_epochs')),  # Number of training epochs
        learning_rate=float(training_config.get('learning_rate')),  # Learning rate
        warmup_ratio=float(training_config.get('warmup_ratio')),  # Warmup ratio
        lr_scheduler_type=str(training_config.get('lr_scheduler_type')),  # Learning rate scheduler type

        # 3. Mixed precision and hardware related
        bf16=bool(training_config.get('bf16')),  # Whether to use bfloat16
        bf16_full_eval=bool(training_config.get('bf16_full_eval', training_config.get('bf16', True))),  # Whether to use bf16 during evaluation
        tf32=bool(training_config.get('tf32', True)),  # Whether to use tf32
        dataloader_pin_memory=bool(training_config.get('dataloader_pin_memory', False)),  # Whether dataloader pins memory
        dataloader_num_workers=int(training_config.get('dataloader_num_workers')),  # Number of dataloader workers
        dataloader_persistent_workers=bool(training_config.get('dataloader_persistent_workers')),  # Whether dataloader workers are persistent
        dataloader_prefetch_factor=int(training_config.get('dataloader_prefetch_factor')),  # Dataloader prefetch factor

        # 4. Evaluation and saving related
        eval_strategy=training_config.get('eval_strategy'),  # Evaluation strategy
        eval_steps=int(training_config.get('eval_steps')),  # Evaluation steps
        eval_delay=int(training_config.get('eval_delay')),  # Evaluation delay
        eval_accumulation_steps=int(training_config.get('eval_accumulation_steps')),  # Evaluation accumulation steps
        logging_steps=int(training_config.get('logging_steps')),  # Logging steps
        save_strategy=training_config.get('save_strategy'),  # Saving strategy
        save_steps=int(training_config.get('save_steps')),  # Saving steps
        save_total_limit=int(training_config.get('save_total_limit')),  # Maximum number of saved models
        save_safetensors=True,  # Save in safetensors format
        load_best_model_at_end=bool(training_config.get('load_best_model_at_end', True)),  # Load best model at end of training
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),  # Metric for best model
        greater_is_better=bool(training_config.get('greater_is_better', False)),  # Whether greater metric is better

        # 5. Optimizer and gradient related
        optim=training_config.get('optim', 'adamw_torch_fused'),  # Optimizer type
        adam_beta1=float(training_config.get('adam_beta1', 0.9)),  # Adam beta1
        adam_beta2=float(training_config.get('adam_beta2', 0.999)),  # Adam beta2
        adam_epsilon=float(training_config.get('adam_epsilon', 1e-8)),  # Adam epsilon
        weight_decay=float(training_config.get('weight_decay', 0.0)),  # Weight decay
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),  # Gradient clipping
        remove_unused_columns=False, 
        gradient_checkpointing=False,  # Already enabled manually, do not set again

        # 6. Distributed and DDP related
        ddp_find_unused_parameters=False,  # DDP find unused parameters

        label_names=["labels"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    # trainer.train(resume_from_checkpoint='fnlp/DX+OPEN_LORA1/checkpoint-13000')
    trainer.train()
    torch.cuda.synchronize()
    
    # Save model
    if use_lora:
        # If using LoRA, merge LoRA weights to base model first, then save complete model
        print("Merging LoRA weights to base model...")
        merged_model = model.merge_and_unload()
        
        # Save the merged complete model with updated method
        merged_model.save_pretrained(output_dir, safe_serialization=False)
        print(f"LoRA weights merged and complete model saved to {output_dir}")
    else:
        # If not using LoRA, save complete model
        trainer.save_model(output_dir)
        print(f"Complete model saved to {output_dir}")
    
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Asteroid TTS Instruct Model")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing the training data")
    parser.add_argument("--eval_data_dir", type=str, required=True, help="Directory containing the evaluation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated audio files")
    parser.add_argument("--training_config", type=str, default="finetune/training_config.yaml",
                        help="Path to the training configuration file")
    parser.add_argument("--lora_config", type=str, default="finetune/lora_config.yaml",
                        help="Path to the LoRA configuration file")
    parser.add_argument("--lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) for fine-tuning")
    
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = MODEL_PATH
    elif not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist.")
    if not args.train_data_dir:
        raise ValueError("Training data directory is required.")
    elif not os.path.exists(args.train_data_dir):
        raise ValueError(f"Training data directory '{args.train_data_dir}' does not exist.")

    if not args.eval_data_dir:
        raise ValueError("Evaluation data directory is required.")
    elif not os.path.exists(args.eval_data_dir):
        raise ValueError(f"Evaluation data directory '{args.eval_data_dir}' does not exist.")
    if not args.output_dir:
        raise ValueError("Output directory is required.")
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # If directory exists, ensure it's writable
        if not os.access(args.output_dir, os.W_OK):
            raise ValueError(f"Output directory '{args.output_dir}' is not writable.")
        print(f"Output directory '{args.output_dir}' already exists and will be used.")

    training_config = {}
    if args.training_config:
        import yaml
        if os.path.exists(args.training_config):
            with open(args.training_config, 'r') as f:
                training_config = yaml.safe_load(f)
            print(f"Successfully loaded training configuration from {args.training_config}: {training_config}")
        else:
            print(f"Warning: Configuration file {args.training_config} does not exist, using default parameters.")
    
    lora_cfg = {}
    if args.lora and args.lora_config:
        import yaml
        if os.path.exists(args.lora_config):
            with open(args.lora_config, 'r') as f:
                lora_cfg = yaml.safe_load(f)
            print(f"Successfully loaded LoRA configuration from {args.lora_config}: {lora_cfg}")
        else:
            print(f"Warning: LoRA configuration file {args.lora_config} does not exist, using default LoRA parameters.")
    
    if args.lora:
        print("Using LoRA fine-tuning mode")
    else:
        print("Using full model fine-tuning mode")
    
    train(args.model_path, args.train_data_dir, args.eval_data_dir, args.output_dir, training_config, device="cuda" if torch.cuda.is_available() else "cpu", use_lora=args.lora, lora_cfg=lora_cfg)