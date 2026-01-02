# Hugging Face Pipelines - Complete Learning Guide

A comprehensive guide to understanding and using Hugging Face Pipelines for LLM/AI Engineering.

## 📚 What is a Pipeline?

A **Pipeline** in Hugging Face is a high-level abstraction that wraps together:

1. **Tokenizer** - Converts raw text into tokens the model can understand
2. **Model** - The pre-trained transformer model that performs the actual inference
3. **Post-processor** - Converts model outputs back into human-readable format

Think of it as a **"plug-and-play" interface** that handles all the complexity of:
- Loading the right model and tokenizer
- Preprocessing your input data
- Running inference
- Post-processing the results

### Why Use Pipelines?

| Without Pipeline | With Pipeline |
|------------------|---------------|
| Load tokenizer manually | Single function call |
| Tokenize input | Automatic preprocessing |
| Convert to tensors | Handled internally |
| Run model inference | Abstracted away |
| Decode outputs | Automatic post-processing |
| ~15-20 lines of code | ~3 lines of code |

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PIPELINE                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  TOKENIZER  │───▶│    MODEL    │───▶│ POST-PROCESSOR  │  │
│  │             │    │             │    │                 │  │
│  │ - Encode    │    │ - Forward   │    │ - Decode        │  │
│  │ - Truncate  │    │ - Inference │    │ - Format        │  │
│  │ - Pad       │    │             │    │ - Aggregate     │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│        ▲                                       │            │
│        │                                       ▼            │
│   Raw Input                              Structured Output  │
│   (text, image,                         (labels, scores,    │
│    audio, etc.)                          text, etc.)        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
huggingface-pipelines-guide/
├── README.md                           # This file
├── QUICKSTART.md                       # Quick start guide
├── notebooks/
│   ├── 01_pipeline_basics.ipynb        # Core concepts and basic usage
│   ├── 02_nlp_pipelines.ipynb          # All NLP pipeline types
│   ├── 03_multimodal_pipelines.ipynb   # Vision, audio, multimodal
│   ├── 04_advanced_pipelines.ipynb     # Custom pipelines, optimization
│   └── 05_production_patterns.ipynb    # Best practices for production
├── examples/
│   ├── sentiment_analysis.py           # Sentiment analysis example
│   ├── text_generation.py              # Text generation example
│   ├── question_answering.py           # QA example
│   ├── summarization.py                # Summarization example
│   └── custom_pipeline.py              # Custom pipeline example
└── requirements.txt                    # Dependencies
```

## 🚀 Quick Start

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Use it!
result = classifier("I love learning about AI!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

## 📋 Pipeline Types Overview

### Natural Language Processing (NLP)

| Pipeline Type | Task | Example Use Case |
|---------------|------|------------------|
| `text-classification` | Classify text into categories | Sentiment analysis, spam detection |
| `token-classification` | Label each token | NER, POS tagging |
| `question-answering` | Extract answers from context | Customer support, search |
| `fill-mask` | Predict masked words | Autocomplete, language modeling |
| `summarization` | Condense long text | News summarization, TL;DR |
| `translation` | Convert between languages | Localization |
| `text-generation` | Generate continuation | Chatbots, creative writing |
| `text2text-generation` | Transform text | Paraphrasing, style transfer |
| `zero-shot-classification` | Classify without training | Flexible categorization |
| `feature-extraction` | Get embeddings | Semantic search, clustering |

### Computer Vision

| Pipeline Type | Task | Example Use Case |
|---------------|------|------------------|
| `image-classification` | Classify images | Photo organization |
| `object-detection` | Locate objects | Autonomous vehicles |
| `image-segmentation` | Pixel-level classification | Medical imaging |
| `depth-estimation` | Estimate depth | AR/VR applications |
| `image-to-text` | Describe images | Accessibility |

### Audio

| Pipeline Type | Task | Example Use Case |
|---------------|------|------------------|
| `automatic-speech-recognition` | Speech to text | Transcription |
| `audio-classification` | Classify audio | Music genre detection |
| `text-to-speech` | Text to audio | Voice assistants |

### Multimodal

| Pipeline Type | Task | Example Use Case |
|---------------|------|------------------|
| `document-question-answering` | QA on documents | Invoice processing |
| `visual-question-answering` | QA on images | Image search |
| `image-to-text` | Image captioning | Accessibility |

## 📖 Learning Path

1. **Start Here**: [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
2. **Basics**: [01_pipeline_basics.ipynb](notebooks/01_pipeline_basics.ipynb) - Core concepts
3. **NLP Deep Dive**: [02_nlp_pipelines.ipynb](notebooks/02_nlp_pipelines.ipynb) - All NLP tasks
4. **Beyond Text**: [03_multimodal_pipelines.ipynb](notebooks/03_multimodal_pipelines.ipynb) - Vision & audio
5. **Advanced**: [04_advanced_pipelines.ipynb](notebooks/04_advanced_pipelines.ipynb) - Custom pipelines
6. **Production**: [05_production_patterns.ipynb](notebooks/05_production_patterns.ipynb) - Best practices

## 🛠️ Installation

```bash
# Basic installation
pip install transformers

# With PyTorch (recommended)
pip install transformers torch

# With TensorFlow
pip install transformers tensorflow

# Full installation with all dependencies
pip install transformers[torch] datasets accelerate
```

## 🔑 Key Concepts to Understand

### 1. Task Abstraction
Pipelines abstract specific ML tasks. You don't need to know the model architecture.

### 2. Model Hub Integration
Pipelines automatically download models from the [Hugging Face Hub](https://huggingface.co/models).

### 3. Device Management
Pipelines handle CPU/GPU placement automatically (or you can specify).

### 4. Batching
Pipelines can process multiple inputs efficiently.

### 5. Tokenization Handling
All text preprocessing is handled internally.

## 📚 Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Model Hub](https://huggingface.co/models)
- [Hugging Face Course](https://huggingface.co/course)
- [Transformers GitHub](https://github.com/huggingface/transformers)

## 📄 License

This learning guide is provided for educational purposes. The code examples use the Apache 2.0 licensed Transformers library.

---

**Happy Learning!** 🤗

*Created for Ola's LLM/AI Engineering learning journey*
