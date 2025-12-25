"""Example of async usage with VertexAI provider."""

import asyncio
import os
from pydantic import BaseModel, Field
from structured_llm import StructuredLLMClient


class SentimentAnalysis(BaseModel):
    """Structured output for sentiment analysis."""
    
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_phrases: list[str] = Field(description="Important phrases that indicate sentiment")
    emotions: list[str] = Field(description="Detected emotions like joy, anger, sadness")


async def analyze_text(client: StructuredLLMClient, text: str, text_id: int) -> tuple[int, SentimentAnalysis]:
    """Analyze a single text asynchronously."""
    print(f"[{text_id}] Starting analysis...")
    
    result = await client.acomplete(
        response_model=SentimentAnalysis,
        user_prompt=f"Analyze the sentiment of this text:\n\n{text}",
        system_prompt="You are an expert at sentiment analysis. Be precise and thorough.",
    )
    
    print(f"[{text_id}] ✓ Complete - Sentiment: {result.sentiment} ({result.confidence:.1%})")
    return text_id, result


async def main():
    """Main async function demonstrating concurrent analysis."""
    
    api_key = os.getenv("VERTEX_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Please set VERTEX_AI_API_KEY or GOOGLE_API_KEY environment variable")
        return
    
    print("Initializing VertexAI provider for async operations...")
    client = StructuredLLMClient(
        provider="vertexai",
        temperature=0.2,
    )
    
    texts = [
        "This product exceeded my expectations! The quality is amazing and shipping was fast.",
        "Terrible experience. The item arrived damaged and customer service was unhelpful.",
        "It's okay, nothing special. Does what it's supposed to do but not impressive.",
        "Absolutely love it! Best purchase I've made this year. Highly recommend!",
        "Not worth the price. Quality is mediocre and there are better alternatives.",
    ]
    
    print(f"\nAnalyzing {len(texts)} texts concurrently...")
    print("=" * 80)
    
    try:
        tasks = [analyze_text(client, text, i+1) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for text_id, analysis in results:
            print(f"\n[Text {text_id}]")
            print(f"  Text: {texts[text_id-1][:60]}...")
            print(f"  Sentiment: {analysis.sentiment.upper()}")
            print(f"  Confidence: {analysis.confidence:.1%}")
            print(f"  Key Phrases: {', '.join(analysis.key_phrases[:3])}")
            print(f"  Emotions: {', '.join(analysis.emotions)}")
            
            if analysis.sentiment.lower() == "positive":
                positive_count += 1
            elif analysis.sentiment.lower() == "negative":
                negative_count += 1
            else:
                neutral_count += 1
        
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        print(f"Total Analyzed: {len(results)}")
        print(f"Positive: {positive_count}")
        print(f"Negative: {negative_count}")
        print(f"Neutral: {neutral_count}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check your API configuration and network connection.")


if __name__ == "__main__":
    asyncio.run(main())

