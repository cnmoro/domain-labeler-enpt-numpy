# Domain Labeler EN+PT — Pure NumPy Inference

Zero-dependency domain classifier for English and Portuguese text. Classifies text into 67 Wikipedia domain categories.

**85% test accuracy** across EN+PT (joint training, no per-language switching). Model size: **~125 MB**.

## How it works

Two frozen embedding models feed into a trainable 2-stream MLP:

| Model | Output | Role |
|-------|--------|------|
| [`cnmoro/inference-free-splade-co-condenser-en-ptbr-v2`](https://huggingface.co/cnmoro/inference-free-splade-co-condenser-en-ptbr-v2) | ~80 active dims over 30,522 vocab | Sparse lexical features |
| [`cnmoro/static-nomic-384-pten-v2`](https://huggingface.co/cnmoro/static-nomic-384-pten-v2) | 384-dim dense | Semantic embeddings |

The SPLADE 30,522-dim vector is **never materialized** — projection uses only the ~80 non-zero dimensions via weight-matrix column indexing. This keeps memory and CPU usage extremely low.

## Usage

```python
from inference import DomainLabeler

model = DomainLabeler(".")
model.predict("O Google Chrome é um navegador web")
# → ["computer_science_and_technology"]

model.predict_proba("Michael Jordan played basketball")
# → {"sports": 0.92, "biography": 0.04, ...}
```

## Dependencies

| Package | Size |
|---------|------|
| `numpy` | ~15 MB |
| `tokenizers` | ~5 MB |
| **Total** | **~20 MB** |

No PyTorch, no Transformers, no ONNX Runtime, no CUDA.

## Docker

```dockerfile
FROM python:3.12-slim
RUN pip install numpy tokenizers
COPY onnx_deploy/ /model/
CMD ["python", "/model/inference.py"]
```

Image size: **~265 MB** (vs. 10+ GB with PyTorch/Transformers).

## Folder structure

```
onnx_deploy/
├── inference.py           # DomainLabeler class
├── requirements.txt       # numpy, tokenizers
├── README.md
└── assets/
    ├── splade_tokenizer.json
    ├── splade_weights.npy
    ├── nomic_tokenizer.json
    ├── nomic_embeddings.npy   # (32000, 384)
    ├── nomic_mapping.npy      # (276214,)
    ├── nomic_weights.npy      # (276214,)
    ├── mlp_splade_proj_weight.npy  # (512, 30522)
    ├── mlp_*.npy               # 14 weight files total
    └── id2label.json           # 67 classes
```

## Training

Trained on ~80K Portuguese + ~80K English Wikipedia-sourced texts, merged and shuffled. Architecture: two-stream MLP with separate projection heads for SPLADE and Nomic features, trained via SGD with log loss. No fine-tuning of the embedding models — they remain frozen.

## Categories (67)

```
aerospace                     astronomy           atmospheric_science
automotive                    beauty              biology
celebrity                     chemistry           civil_engineering
communication_engineering     computer_science_and_technology
design                        drama_and_film      economics
electronic_science            entertainment       environmental_science
fashion                       finance             food
gamble                        game                geography
health                        history             hobby
hydraulic_engineering         instrument_science
journalism_and_media_communication                landscape_architecture
law                           library             literature
materials_science             mathematics         mechanical_engineering
medical                       mining_engineering  movie
music_and_dance               news                nuclear_science
ocean_science                 optical_engineering painting
pet                           petroleum_and_natural_gas_engineering
philosophy                    photo               physics
politics                      psychology          public_administration
relationship                  religion            sociology
sports                        statistics          systems_science
textile_science               topicality
transportation_engineering    travel              urban_planning
vulgar_language
```

## Attribution

- Base dataset: [`NeuML/wikipedia-domain-labels`](https://huggingface.co/datasets/NeuML/wikipedia-domain-labels)
- Translated PTBR dataset: [`cnmoro/wikipedia-domain-labels-ptbr`](https://huggingface.co/datasets/cnmoro/wikipedia-domain-labels-ptbr)
- SPLADE encoder: [`cnmoro/inference-free-splade-co-condenser-en-ptbr-v2`](https://huggingface.co/cnmoro/inference-free-splade-co-condenser-en-ptbr-v2)
- Dense encoder: [`cnmoro/static-nomic-384-pten-v2`](https://huggingface.co/cnmoro/static-nomic-384-pten-v2)
