import os
import sys


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils.llm import call_llm

    platforms = ["BAILIAN", "openrouter"]
    models = ["qwen3-8b","openai/gpt-4.1-mini"]
    prompt = "Say hello and identify which platform you are responding from."

    for index, platform in enumerate(platforms):
        model = models[index]
        print(f"\n===== Testing platform: {platform} =====")
        try:
            response = call_llm(
                text=prompt,
                system_prompt="You are a simple echo assistant.",
                model=model,
                max_tokens=64,
                temperature=0.2,
                platform=platform,
            )
            print("Response:")
            print(response)
        except Exception as e:
            print(f"Error while calling platform '{platform}': {e}")


if __name__ == "__main__":
    main()

