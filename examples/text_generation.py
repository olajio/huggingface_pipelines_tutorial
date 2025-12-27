"""
Text Generation Example
=======================
Demonstrates text generation pipeline with various parameters.
"""

from transformers import pipeline


def main():
    # Create pipeline
    print("Loading text generation pipeline...")
    generator = pipeline("text-generation", model="gpt2")
    
    prompt = "The future of artificial intelligence is"
    
    # Basic generation
    print("\n--- Basic Generation ---")
    result = generator(prompt, max_length=50, num_return_sequences=1)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result[0]['generated_text']}")
    
    # With temperature control
    print("\n--- Low Temperature (More Focused) ---")
    result = generator(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.3
    )
    print(result[0]['generated_text'])
    
    print("\n--- High Temperature (More Creative) ---")
    result = generator(
        prompt,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0
    )
    print(result[0]['generated_text'])
    
    # Multiple sequences
    print("\n--- Multiple Sequences ---")
    results = generator(
        prompt,
        max_new_tokens=25,
        num_return_sequences=3,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['generated_text']}")
    
    # With specific stopping
    print("\n--- Controlled Generation ---")
    result = generator(
        "Write a haiku about programming:\n",
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    print(result[0]['generated_text'])


if __name__ == "__main__":
    main()
