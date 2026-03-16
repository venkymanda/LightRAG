# LoRA/SFT for LightRAG Entity-Relation Extraction (Minimal Colab Plan)

This guide gives a practical, low-cost workflow to adapt an open model for LightRAG entity/relation extraction.

It includes:

1. A minimal Google Colab LoRA plan (libraries + concrete training args).
2. A dataset template tailored for LightRAG-style extraction outputs.

## 1) What to train: LoRA vs SFT

- **LoRA**: Train small adapter weights on top of a frozen base model. Best first step for cost/latency-sensitive adaptation.
- **SFT**: Update more/all model weights. Higher resource cost; use only if LoRA quality is insufficient.

For most extraction customization tasks, start with **QLoRA** (4-bit base model + LoRA adapters).

## 2) Minimal Colab plan (exact libs + baseline args)

### Recommended Colab runtime

- GPU runtime with at least 15GB VRAM (T4/L4 works for small-to-mid models).
- Use checkpointing to Google Drive because free Colab sessions can reset.

### Install libraries

```bash
pip install -U "transformers==4.44.2" "datasets==2.21.0" "peft==0.12.0" \
  "trl==0.10.1" "accelerate==0.34.2" "bitsandbytes==0.43.3" \
  "sentencepiece==0.2.0" "safetensors==0.4.4"
```

### Minimal training script settings

Use these as a starting profile for extraction fine-tuning:

- **Base model size**: 3B to 8B instruct model
- **Quantization**: 4-bit (`nf4`, bfloat16 compute)
- **LoRA target modules**: `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`
- **LoRA rank**: `r=16`
- **LoRA alpha**: `32`
- **LoRA dropout**: `0.05`
- **Max sequence length**: `2048`
- **Per-device batch size**: `1`
- **Gradient accumulation**: `16`
- **Learning rate**: `2e-4`
- **Scheduler**: `cosine`
- **Warmup ratio**: `0.03`
- **Epochs**: `2`
- **Weight decay**: `0.01`
- **Gradient checkpointing**: `true`
- **Logging steps**: `10`
- **Save steps**: `200`

These defaults are intentionally conservative for free/low-tier Colab stability.

### Example training stub (TRL SFTTrainer)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Replace with your preferred model
DATA_PATH = "/content/lightrag_extraction_train.jsonl"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

args = TrainingArguments(
    output_dir="/content/lightrag_extraction_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
    args=args,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()
trainer.model.save_pretrained("/content/lightrag_extraction_lora/final")
tokenizer.save_pretrained("/content/lightrag_extraction_lora/final")
```

## 3) Dataset template for LightRAG extraction

### Recommended JSONL format

Use one training sample per line with a `text` field containing instruction + input + target output.

```json
{"text":"<|system|>You extract entities and relationships from text. Return strict JSON only.<|user|>Input text:\nAlice founded Acme in 2020 in Seattle. Bob joined Acme in 2021 as CTO.<|assistant|>{\"entities\":[{\"name\":\"Alice\",\"type\":\"person\",\"description\":\"Founder of Acme\"},{\"name\":\"Acme\",\"type\":\"organization\",\"description\":\"Company founded in 2020\"},{\"name\":\"Seattle\",\"type\":\"location\",\"description\":\"City where Acme was founded\"},{\"name\":\"Bob\",\"type\":\"person\",\"description\":\"Joined Acme as CTO\"}],\"relationships\":[{\"source\":\"Alice\",\"target\":\"Acme\",\"relation\":\"founded\",\"description\":\"Alice founded Acme\",\"keywords\":[\"founder\",\"company\"],\"weight\":1.0},{\"source\":\"Acme\",\"target\":\"Seattle\",\"relation\":\"located_in\",\"description\":\"Acme was founded in Seattle\",\"keywords\":[\"headquarters\",\"location\"],\"weight\":0.8},{\"source\":\"Bob\",\"target\":\"Acme\",\"relation\":\"works_at\",\"description\":\"Bob joined Acme as CTO\",\"keywords\":[\"employment\",\"cto\"],\"weight\":0.9}],\"source_text_id\":\"doc_001_chunk_001\"}"}
```

### Structured target schema

Keep the output schema stable across the whole dataset:

```json
{
  "entities": [
    {
      "name": "string",
      "type": "person|organization|location|event|concept|...",
      "description": "string"
    }
  ],
  "relationships": [
    {
      "source": "entity_name",
      "target": "entity_name",
      "relation": "verb_or_relation_label",
      "description": "string",
      "keywords": ["string", "string"],
      "weight": 0.0
    }
  ],
  "source_text_id": "string"
}
```

### Data quality checklist

- Use the same entity typing policy in every sample.
- Include hard negatives (no relation, weak evidence, pronoun ambiguity).
- Include multilingual/domain-specific examples you expect in production.
- Keep outputs valid JSON (no markdown wrappers).
- Deduplicate near-identical samples to reduce overfitting.

## 4) Train/validation split suggestion

- Train: 80%
- Validation: 10%
- Test: 10%

For small datasets, prioritize diversity over size.

## 5) Inference prompt template (post-training)

Use a strict instruction format during serving to reduce formatting drift:

```text
You extract entities and relationships from the user text.
Return strict JSON with keys: entities, relationships, source_text_id.
Do not add markdown fences, comments, or extra keys.
```

## 6) Free Colab feasibility notes

- Good for prototyping LoRA and checking extraction quality.
- Not ideal for long-running training jobs or high-parameter full SFT.
- Save checkpoints frequently and resume training when sessions reset.

## 7) Production progression

1. Start with prompt-only baseline.
2. Apply LoRA with the template above.
3. Evaluate extraction precision/recall on held-out corpus.
4. Iterate dataset quality before increasing model size.
5. Move to paid GPU or managed training for stable large-scale runs.
