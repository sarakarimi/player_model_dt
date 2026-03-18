# Style Decision Transformer

## Requirements
```
python 3.10.6
```

## Setup
Upgrade pip (we recommend creating a virtual environment for this project)
```
pip install --upgrade pip
```
Install dependencies:
```
pip install -r requirements.txt
```


## Hyperparameters and training details 

**Architecture**

  | Hyperparameter | StyleDT (ours) | ControlDT | PromptDT | BC | SORL |
  |---|---|---|---|---|---|
  | Hidden dim | 128 | 128 | 128 | 256 | 256 |
  | Transformer layers | 4 | 4 | 4 | — | — |
  | Attention heads | 8 | 8 | 8 | — | — |
  | MLP layers | — | — | — | 3 | 3 |
  | Context window | 8 | 8 | 8 | — | — |
  | Prompt length | — | — | 2 | — | — |
  | Latent dim | 16 | — | — | — | — |
  | Prior hidden dim | 128 | — | — | — | — |
  | Control dim | 3 | 3 | — | — | — |

**Training**

 | Hyperparameter | StyleDT (ours) | ControlDT | PromptDT | BC | SORL |
  |---|---|---|---|---|------|
  | Batch size | 32 | 32 | 32 | 256 | 256  |
  | Epochs | 100 | 100 | 100 | 100 | 100  |
  | Learning rate | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
  | Gradient clip | 1.0 | 1.0 | 1.0 | — | 1.0  |

**Method-specific**

| Hyperparameter | StyleDT (ours) | ControlDT | PromptDT | BC | SORL |
|---|---|---|---|---|---|
  | KL weight β | 0.0085 | — | — | — | — |
  | β warmup epochs | 20 | — | — | — | — |
  | EM iterations | — | — | — | — | 30 |
  | M-step epochs/iter | — | — | — | — | 5 |
  | BC warmup epochs | — | — | — | — | 5 |
  | Advantage weight β | — | — | — | — | 1.0 |
  | Advantage clip | — | — | — | — | 5.0 |
  | EM temperature τ | — | — | — | — | 1.0 |
  | Value net epochs | — | — | — | — | 100 |
