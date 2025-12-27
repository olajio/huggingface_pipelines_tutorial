"""
Sentiment Analysis Example
==========================
Demonstrates basic sentiment analysis pipeline usage.
"""

from transformers import pipeline


def main():
    # Create pipeline
    print("Loading sentiment analysis pipeline...")
    classifier = pipeline("sentiment-analysis")
    
    # Single text analysis
    print("\n--- Single Text Analysis ---")
    text = "I absolutely love learning about transformers and NLP!"
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Result: {result[0]['label']} (confidence: {result[0]['score']:.4f})")
    
    # Batch analysis
    print("\n--- Batch Analysis ---")
    texts = [
        "This product is amazing! Best purchase ever.",
        "Terrible quality. I want a refund.",
        "It's okay, nothing special.",
        "The customer service was excellent!",
        "Disappointed with the shipping time."
    ]
    
    results = classifier(texts)
    
    for text, result in zip(texts, results):
        emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
        print(f"{emoji} {result['label']:8} ({result['score']:.2%}): {text[:50]}...")
    
    # Using a different model (5-star ratings)
    print("\n--- 5-Star Rating Model ---")
    rating_classifier = pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    
    review = "The product quality is excellent but delivery was slow."
    result = rating_classifier(review)[0]
    print(f"Review: {review}")
    print(f"Rating: {result['label']} (confidence: {result['score']:.4f})")


if __name__ == "__main__":
    main()
