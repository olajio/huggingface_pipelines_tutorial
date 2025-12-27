"""
Custom Pipeline Example
=======================
Demonstrates how to create custom pipelines for specialized tasks.
"""

from transformers import Pipeline, pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class SentimentWithEmoji(Pipeline):
    """
    Custom pipeline that adds emoji to sentiment results.
    """
    
    def _sanitize_parameters(self, **kwargs):
        """Process any custom parameters."""
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        if "return_emoji" in kwargs:
            postprocess_kwargs["return_emoji"] = kwargs["return_emoji"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, text):
        """Tokenize the input text."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
    
    def _forward(self, model_inputs):
        """Run the model."""
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs, return_emoji=True):
        """Process model outputs into final results."""
        logits = model_outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Map to label
        label = self.model.config.id2label[predicted_class]
        
        result = {
            "label": label,
            "score": confidence,
            "confidence_level": "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
        }
        
        if return_emoji:
            emoji_map = {
                "POSITIVE": "😊",
                "NEGATIVE": "😞",
                "NEUTRAL": "😐"
            }
            result["emoji"] = emoji_map.get(label.upper(), "🤔")
        
        return result


class TextAnalyzer:
    """
    Combines multiple pipelines for comprehensive text analysis.
    """
    
    def __init__(self):
        print("Loading models...")
        self.sentiment = pipeline("sentiment-analysis")
        self.ner = pipeline("ner", grouped_entities=True)
        self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    def analyze(self, text: str, topic_labels: list = None) -> dict:
        """
        Perform comprehensive text analysis.
        """
        results = {
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
        
        # Sentiment analysis
        sentiment = self.sentiment(text)[0]
        results["sentiment"] = {
            "label": sentiment["label"],
            "score": round(sentiment["score"], 4)
        }
        
        # Named entity recognition
        entities = self.ner(text)
        results["entities"] = [
            {
                "text": e["word"],
                "type": e["entity_group"],
                "score": round(e["score"], 4)
            }
            for e in entities
        ]
        
        # Topic classification (if labels provided)
        if topic_labels:
            topics = self.zero_shot(text, topic_labels)
            results["topics"] = {
                label: round(score, 4)
                for label, score in zip(topics["labels"], topics["scores"])
            }
        
        return results


def demo_custom_pipeline():
    """Demonstrate the custom sentiment pipeline."""
    print("\n" + "="*60)
    print("CUSTOM PIPELINE DEMO")
    print("="*60)
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create custom pipeline
    custom_pipe = SentimentWithEmoji(
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
    
    # Test texts
    texts = [
        "This is absolutely fantastic!",
        "I'm very disappointed with the service.",
        "The weather is okay today."
    ]
    
    print("\n--- Custom Sentiment Analysis ---")
    for text in texts:
        result = custom_pipe(text)
        print(f"\n{result['emoji']} {text}")
        print(f"   Label: {result['label']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Confidence: {result['confidence_level']}")


def demo_text_analyzer():
    """Demonstrate the combined text analyzer."""
    print("\n" + "="*60)
    print("TEXT ANALYZER DEMO")
    print("="*60)
    
    analyzer = TextAnalyzer()
    
    text = """
    Apple CEO Tim Cook announced today that the company will invest $1 billion 
    in artificial intelligence research at their headquarters in Cupertino, California. 
    The initiative is expected to create 500 new jobs over the next three years.
    """
    
    topic_labels = ["technology", "business", "politics", "sports", "entertainment"]
    
    results = analyzer.analyze(text.strip(), topic_labels)
    
    print(f"\nText: {results['text'][:80]}...")
    print(f"\nWord count: {results['word_count']}")
    print(f"Sentiment: {results['sentiment']['label']} ({results['sentiment']['score']:.2%})")
    
    print("\nEntities found:")
    for entity in results['entities']:
        print(f"  - {entity['type']}: {entity['text']} ({entity['score']:.2%})")
    
    print("\nTopic scores:")
    for topic, score in sorted(results['topics'].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"  {topic:15} {bar} {score:.2%}")


def main():
    demo_custom_pipeline()
    demo_text_analyzer()


if __name__ == "__main__":
    main()
