# Assignment 1 — Safe Generative AI: Reward Modelling, PPO and DPO Fine-Tuning

RLHF-style alignment pipeline for GPT-2 Medium, comparing reward-model-based PPO against reference-free DPO across multiple PEFT strategies.

## What's in here

| File | Description |
|---|---|
| `Task_A_reward.ipynb` | Trains a BERT-based reward model (`bert-base-uncased` + linear head on `[CLS]`) on 5K pairwise preference samples using Bradley-Terry loss. |
| `Task_A_alignment.ipynb` | Fine-tunes GPT-2 Medium via PPO (TRL) using the trained reward model, across Full, LoRA, and QLoRA strategies. |
| `Task_B.ipynb` | Fine-tunes GPT-2 Medium via DPO (no reward model needed) across Full, Prefix, LoRA, and QLoRA. |
| `Task_C.ipynb` | Evaluation of all HuggingFace-hosted checkpoints on 6,000 held-out preference pairs (BLEU, ROUGE-L, semantic similarity). |
| `Report.pdf` | Full writeup: methodology, hyperparameters, results, and failure analysis. |

## Key results

| Strategy | BLEU | ROUGE-L | Sem-Sim |
|---|---|---|---|
| Baseline (human reference) | 0.0259 | 0.1681 | 0.6250 |
| PPO – Full | 0.0000 | 0.0020 | 0.0363 |
| PPO – LoRA | 0.0000 | 0.0022 | 0.0344 |
| PPO – QLoRA | 0.0008 | 0.0603 | 0.2994 |
| **DPO – Full** | 0.0021 | 0.0764 | **0.4391** |
| DPO – LoRA | 0.0014 | 0.0863 | 0.2644 |
| DPO – QLoRA | 0.0014 | 0.0825 | 0.2808 |

**DPO consistently outperforms PPO.** PPO-Full and PPO-LoRA collapsed to near-zero scores — the policy overfit to gaming the reward model rather than producing coherent text. QLoRA's quantization acted as implicit regularization, keeping PPO more stable than its Full/LoRA counterparts (Sem-Sim 0.2994 vs. ~0.035).

**Notable engineering issue:** Prefix Tuning was infeasible under PPO due to a TRL 0.8.6 `isinstance` check incompatible with PEFT-wrapped prefix models — documented in the report along with the workarounds attempted (custom `PreTrainedModelWrapper` subclass, manual architecture registration) and why they didn't resolve it without a transformers downgrade.

## Checkpoints

All trained models are on HuggingFace: [huggingface.co/mysteriousgirl](https://huggingface.co/mysteriousgirl)

- Reward Model (BERT): `mysteriousgirl/reward-model-bert`
- PPO: `ppo-gpt2-full`, `ppo-gpt2-lora`, `ppo-gpt2-qlora`
- DPO: `dpo-gpt2-full`, `dpo-gpt2-prefix`, `dpo-gpt2-lora`, `dpo-gpt2-qlora`

## Reproducibility

- Platforms: Google Colab (reward model, PPO), Kaggle GPU T4 (DPO)
- `transformers==4.40.0`, `trl==0.8.6`, `peft==0.10.0`
- Seed: `random_state=42` for all sampling, 5,000 training samples
