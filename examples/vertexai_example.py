"""Example usage of the VertexAI provider with structured-llm."""

import os
from pydantic import BaseModel, Field
from structured_llm import StructuredLLMClient


class RecipeAnalysis(BaseModel):
    """Structured output model for recipe analysis."""
    
    cuisine: str = Field(description="The cuisine type (e.g., Italian, Chinese, French)")
    difficulty: str = Field(description="Difficulty level: Easy, Medium, or Hard")
    cooking_time_minutes: int = Field(description="Estimated cooking time in minutes")
    dietary_tags: list[str] = Field(description="Dietary tags like vegetarian, vegan, gluten-free")
    main_ingredients: list[str] = Field(description="List of main ingredients")
    confidence: float = Field(description="Confidence score between 0 and 1")


def main():
    """Main function to demonstrate VertexAI provider usage."""
    
    api_key = os.getenv("VERTEX_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Please set VERTEX_AI_API_KEY or GOOGLE_API_KEY environment variable")
        print("\nExample:")
        print("  export VERTEX_AI_API_KEY='your-api-key-here'")
        return
    
    print("Initializing VertexAI provider...")
    client = StructuredLLMClient(
        provider="vertexai",
        temperature=0.1,
        max_retries=3,
    )
    
    recipe_text = """
    Pasta Carbonara
    
    Ingredients:
    - 400g spaghetti
    - 200g pancetta or guanciale
    - 4 large eggs
    - 100g Pecorino Romano cheese
    - Black pepper
    - Salt
    
    Instructions:
    1. Cook the spaghetti in salted boiling water
    2. Meanwhile, fry the pancetta until crispy
    3. Beat eggs with grated cheese and black pepper
    4. Drain pasta and mix with pancetta
    5. Remove from heat and quickly stir in egg mixture
    6. The residual heat will cook the eggs into a creamy sauce
    
    Time: About 20 minutes
    """
    
    print("\nAnalyzing recipe...")
    print("-" * 60)
    
    try:
        result = client.complete(
            response_model=RecipeAnalysis,
            user_prompt=f"Analyze this recipe and extract structured information:\n\n{recipe_text}",
            system_prompt="You are a professional chef and recipe analyzer. Provide accurate analysis.",
        )
        
        print("\n✓ Analysis Complete!")
        print("=" * 60)
        print(f"Cuisine:        {result.cuisine}")
        print(f"Difficulty:     {result.difficulty}")
        print(f"Cooking Time:   {result.cooking_time_minutes} minutes")
        print(f"Dietary Tags:   {', '.join(result.dietary_tags)}")
        print(f"Main Ingredients:")
        for ingredient in result.main_ingredients:
            print(f"  - {ingredient}")
        print(f"Confidence:     {result.confidence:.2%}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check:")
        print("  1. Your API key is valid")
        print("  2. You have sufficient quota")
        print("  3. Your internet connection is working")


if __name__ == "__main__":
    main()

