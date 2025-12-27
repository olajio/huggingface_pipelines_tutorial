"""
Question Answering Example
==========================
Demonstrates extractive QA pipeline usage.
"""

from transformers import pipeline


def main():
    # Create pipeline
    print("Loading question answering pipeline...")
    qa = pipeline("question-answering")
    
    # Context about a company
    context = """
    Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf 
    originally as a company building a chatbot app. After open-sourcing the model behind 
    the chatbot, the company pivoted to focus on machine learning. The company is 
    headquartered in New York City. They developed the popular Transformers library, 
    which provides thousands of pretrained models for natural language processing tasks.
    As of 2023, the Hugging Face Hub hosts over 200,000 models and 30,000 datasets.
    The company has raised over $235 million in funding and is valued at $4.5 billion.
    """
    
    # Ask questions
    questions = [
        "When was Hugging Face founded?",
        "Who founded Hugging Face?",
        "Where is the company headquartered?",
        "What library did they develop?",
        "How many models are on the Hub?",
        "What is the company's valuation?"
    ]
    
    print("\n" + "="*60)
    print("QUESTION ANSWERING DEMO")
    print("="*60)
    print(f"\nContext: {context[:100]}...")
    print("\n" + "-"*60)
    
    for question in questions:
        result = qa(question=question, context=context)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"   Confidence: {result['score']:.2%}")
        print(f"   Position: chars {result['start']}-{result['end']}")
    
    # Get multiple answers
    print("\n" + "-"*60)
    print("\n--- Top 3 Answers for a Question ---")
    
    result = qa(
        question="What did the company originally build?",
        context=context,
        top_k=3
    )
    
    for i, ans in enumerate(result, 1):
        print(f"{i}. {ans['answer']} (score: {ans['score']:.4f})")


if __name__ == "__main__":
    main()
