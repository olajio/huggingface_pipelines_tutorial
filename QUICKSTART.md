# 🚀 Quick Start Guide - Hugging Face Pipelines

Get up and running with Hugging Face Pipelines in 5 minutes.

## Installation

```bash
pip install transformers torch
```

## Your First Pipeline

```python
from transformers import pipeline

# Create a pipeline - it downloads the model automatically!
classifier = pipeline("sentiment-analysis")

# Analyze sentiment
result = classifier("This tutorial is amazing!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Common Pipeline Patterns

### Pattern 1: Default Model

```python
# Uses default model for the task
pipe = pipeline("text-classification")
```

### Pattern 2: Specific Model

```python
# Use a specific model from the Hub
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
```

### Pattern 3: GPU Acceleration

```python
# Run on GPU (device=0 for first GPU)
pipe = pipeline("text-generation", device=0)

# Or use device_map for automatic placement
pipe = pipeline("text-generation", device_map="auto")
```

### Pattern 4: Batch Processing

```python
texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, I guess."
]
results = classifier(texts)  # Process all at once
```

## 5 Essential Pipelines to Know

### 1. Sentiment Analysis

```python
classifier = pipeline("sentiment-analysis")
classifier("I'm having a great day!")
# [{'label': 'POSITIVE', 'score': 0.9999}]
```

### 2. Text Generation

```python
generator = pipeline("text-generation", model="gpt2")
generator("The future of AI is", max_length=50)
# [{'generated_text': 'The future of AI is...'}]
```

### 3. Question Answering

```python
qa = pipeline("question-answering")
qa(question="What is the capital of France?",
   context="France is a country in Europe. Paris is the capital of France.")
# {'answer': 'Paris', 'score': 0.99}
```

### 4. Named Entity Recognition

```python
ner = pipeline("ner", grouped_entities=True)
ner("My name is Ola and I work at HedgeServ in Texas.")
# [{'entity_group': 'PER', 'word': 'Ola', ...},
#  {'entity_group': 'ORG', 'word': 'HedgeServ', ...},
#  {'entity_group': 'LOC', 'word': 'Texas', ...}]
```

### 5. Summarization

```python
summarizer = pipeline("summarization")
summarizer(long_article, max_length=100, min_length=30)
# [{'summary_text': '...'}]
```

## Quick Reference Card

| Task | Pipeline Name | Input | Output |
|------|---------------|-------|--------|
| Sentiment | `sentiment-analysis` | Text | Label + Score |
| NER | `ner` | Text | Entities |
| QA | `question-answering` | Question + Context | Answer |
| Summarize | `summarization` | Long text | Short text |
| Generate | `text-generation` | Prompt | Continuation |
| Translate | `translation_xx_to_yy` | Source text | Target text |
| Fill Mask | `fill-mask` | Text with [MASK] | Predictions |

## Common Parameters

```python
pipeline(
    task="text-classification",      # Required: the task type
    model="model-name",              # Optional: specific model
    tokenizer="tokenizer-name",      # Optional: specific tokenizer
    device=0,                        # Optional: GPU device ID
    device_map="auto",               # Optional: automatic device placement
    torch_dtype=torch.float16,       # Optional: use half precision
    batch_size=8,                    # Optional: batch size for inference
)
```

## Troubleshooting

### Model Too Large?

```python
# Use a smaller model
pipe = pipeline("text-generation", model="distilgpt2")

# Or use quantization
pipe = pipeline("text-generation", model="gpt2", torch_dtype=torch.float16)
```

### Running Out of Memory?

```python
# Process in smaller batches
results = []
for i in range(0, len(texts), 8):
    batch = texts[i:i+8]
    results.extend(pipe(batch))
```

### Slow Performance?

```python
# 1. Use GPU
pipe = pipeline("task", device=0)

# 2. Enable batching
pipe(texts, batch_size=16)

# 3. Use a faster model
pipe = pipeline("task", model="distilbert-base-uncased")
```

## Next Steps

Ready to dive deeper? Check out:

1. [01_pipeline_basics.ipynb](notebooks/01_pipeline_basics.ipynb) - Full tutorial
2. [02_nlp_pipelines.ipynb](notebooks/02_nlp_pipelines.ipynb) - All NLP tasks
3. [README.md](README.md) - Complete documentation

---

**Pro Tip**: The first time you use a model, it downloads from the Hugging Face Hub. Subsequent uses are instant!
