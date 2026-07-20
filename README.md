# Multilingual Health Question-Answering Chatbot for Low-Resource African Languages

Fine-tuning `google/mt5-base` to answer maternal, sexual and reproductive health (MSRH) questions across 7 language-country subsets covering English (Uganda, Ghana, Ethiopia, Kenya), Akan, Luganda, and Swahili.

---

## Results

| Subset | Language | ROUGE-1 | ROUGE-L |
|--------|----------|---------|---------|
| Eng_Eth | English (Ethiopia) | 0.557 | 0.519 |
| Aka_Gha | Akan (Ghana) | 0.379 | 0.249 |
| Eng_Gha | English (Ghana) | 0.367 | 0.262 |
| Eng_Uga | English (Uganda) | 0.323 | 0.254 |
| Swa_Ken | Swahili (Kenya) | 0.319 | 0.223 |
| Eng_Ken | English (Kenya) | 0.276 | 0.191 |
| Lug_Uga | Luganda (Uganda) | 0.188 | 0.140 |

Final validation combined score: **0.2225** (ROUGE-1 × 0.37 + ROUGE-L × 0.37)

---

## Approach

### Model
`google/mt5-base` — a multilingual encoder-decoder transformer pretrained on 101 languages (mC4 corpus), covering all four target African languages. Chosen because:
- Encoder-decoder architecture is the right architecture for open-domain QA generation (unlike encoder-only models like BERT which cannot generate sequences)
- Native multilingual coverage without needing per-language translation pipelines
- Feasible on a free-tier T4 GPU (580M parameters)

### Key design decisions

**Language prefix prompting** — every input is prepended with a native-language instruction before being fed to the model:
```
"Jibu swali hili la afya: <swahili question>"
"ይህንን የጤና ጥያቄ ይመልሱ: <amharic question>"
```
This conditions the model's output language on the input prefix, preventing cross-language generation errors.

**Low-resource augmentation** — `Amh_Eth` (1,845 rows), `Swa_Ken` (2,070), and `Eng_Ken` (2,080) are duplicated during training to balance against `Eng_Uga` (7,624 rows). Over-duplication (3×+) on Amharic was found to hurt precision, so 1× is used.

**Memory-efficient training** — ran on a free Colab T4 (15GB VRAM):
- Adafactor optimizer
- Gradient checkpointing
- `MAX_TARGET_LENGTH = 128`, `MAX_INPUT_LENGTH = 128`
- Batch size 8, gradient accumulation 4 (effective batch = 32)
- bf16 mixed precision
- `model.config.use_cache = False` during training (required with gradient checkpointing)
- `torch_dtype` NOT set at model load time (causes gradient instability with gradient checkpointing)

**Retrieval shortcut at inference** — a lookup dict is built from the training set. If a test question exactly matches a training question (same subset), the training answer is returned directly, bypassing generation. This maximises ROUGE for repeated questions.

**Beam search at inference** — `num_beams=5` with `no_repeat_ngram_size=3` and `length_penalty=1.2`. Greedy decoding (`num_beams=1`) is used during training-time evaluation for speed.

### Bugs encountered and fixed
- `OverflowError` in `compute_metrics`: fp16/bf16 predictions need `.astype(np.int32)` before `batch_decode`
- Flat training loss (~12–17): caused by `torch_dtype=torch.bfloat16` at model load + `use_cache=True` conflicting with gradient checkpointing
- OOM: resolved with Adafactor, gradient checkpointing, and reduced sequence lengths

---

## Repo structure

```
├── Datasets/
│   ├── Train.csv          # training set
    ├── Val.csv            # validation set
│   └── Test.csv           # test set
├── Notebooks/
│   ├── main.ipynb          # training notebook
│   └── inference.ipynb     # inference + full val set evaluation
├── requirements.txt
└── README.md
```

---

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add data
Place `Train.csv`, `Val.csv`, and `Test.csv` in the working directory.

### 3. Train
Open `notebooks/main.ipynb` and run all cells top to bottom.

Checkpoints save to Google Drive at `/content/drive/MyDrive/mt5_health_qa/`. To resume after a Colab disconnect:
```python
trainer.train(resume_from_checkpoint="/content/drive/MyDrive/mt5_health_qa/checkpoint-XXXX")
```

### 4. Inference
Open `notebooks/inference.ipynb`, point `CKPT_DIR` at your saved checkpoint, and run all cells. Outputs `submission.csv`.

---

## Hardware
- GPU: NVIDIA T4 (15GB VRAM) — Google Colab free tier
- Training time: ~6 hours for 6 epochs on the full augmented dataset

## Dependencies
See `requirements.txt`. All packages are free and open-source.
