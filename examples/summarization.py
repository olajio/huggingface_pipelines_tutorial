"""
Summarization Example
=====================
Demonstrates text summarization pipeline usage.
"""

from transformers import pipeline


def main():
    # Create pipeline
    print("Loading summarization pipeline...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Long article to summarize
    article = """
    Artificial intelligence has undergone a remarkable transformation in recent years, 
    fundamentally changing how we interact with technology and process information. 
    The emergence of large language models, particularly those based on transformer 
    architectures, has enabled machines to understand and generate human-like text 
    with unprecedented accuracy.

    These advances have found applications across virtually every industry. In healthcare, 
    AI systems can now analyze medical images with accuracy rivaling experienced 
    radiologists, potentially detecting diseases earlier and improving patient outcomes. 
    Financial institutions employ AI for fraud detection, risk assessment, and 
    algorithmic trading, processing vast amounts of data in milliseconds.

    However, these technological advances come with significant challenges. Concerns 
    about AI bias, job displacement, and the potential misuse of AI-generated content 
    have led to calls for stricter regulation. The European Union has proposed the 
    AI Act, which would establish strict rules for high-risk AI applications, while 
    the United States has taken a more sector-specific approach to regulation.

    Despite these challenges, investment in AI continues to accelerate. Major technology 
    companies and startups alike are racing to develop more powerful and efficient AI 
    systems. The development of multimodal models that can process text, images, and 
    audio simultaneously represents the next frontier in AI research.

    Looking ahead, experts predict that AI will become increasingly integrated into 
    everyday life. From autonomous vehicles to personalized education, AI promises 
    to revolutionize how we live and work. The key challenge will be ensuring that 
    these powerful tools are developed and deployed responsibly, with appropriate 
    safeguards to protect privacy and prevent misuse.
    """
    
    print("\n" + "="*60)
    print("TEXT SUMMARIZATION DEMO")
    print("="*60)
    print(f"\nOriginal article length: {len(article.split())} words")
    
    # Standard summary
    print("\n--- Standard Summary (50-100 words) ---")
    summary = summarizer(article, max_length=100, min_length=50, do_sample=False)
    print(summary[0]['summary_text'])
    print(f"\nSummary length: {len(summary[0]['summary_text'].split())} words")
    
    # Short summary
    print("\n--- Short Summary (30-50 words) ---")
    short_summary = summarizer(article, max_length=50, min_length=30, do_sample=False)
    print(short_summary[0]['summary_text'])
    
    # Longer summary
    print("\n--- Detailed Summary (100-150 words) ---")
    long_summary = summarizer(article, max_length=150, min_length=100, do_sample=False)
    print(long_summary[0]['summary_text'])
    
    # Comparison
    print("\n" + "-"*60)
    print("SUMMARY COMPARISON:")
    print(f"  Original: {len(article.split())} words")
    print(f"  Short:    {len(short_summary[0]['summary_text'].split())} words")
    print(f"  Standard: {len(summary[0]['summary_text'].split())} words")
    print(f"  Detailed: {len(long_summary[0]['summary_text'].split())} words")


if __name__ == "__main__":
    main()
