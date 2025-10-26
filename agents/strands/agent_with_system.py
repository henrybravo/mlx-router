# This script requires mlx_model.py to be in the same directory
# mlx_model.py provides the custom MLXModel class for Strands integration with MLX Router
from mlx_model import MLXModel
import asyncio
from mlx_model import MLXModel
from strands import Agent

async def main():
    # Initialize your custom MLX model
    # mlx_model = MLXModel(
    #     base_url="http://host.docker.internal:8888/v1",
    #     model_id="mlx-community/Qwen3-30B-A3B-8bit",
    #     api_key="strands-key",
    #     params={
    #         "max_tokens": 5000,
    #         "temperature": 0.7,
    #     }
    # )
    mlx_model = MLXModel(
        base_url="http://localhost:8800/v1",
        model_id="mlx-community/gpt-oss-120b-MXFP4-Q8",
        api_key="strands-key",
        params={
            "max_tokens": 16384,
            "temperature": 0.7,
        }
    )

    # Test with system prompt
    agent_with_system = Agent(
        model=mlx_model,
        system_prompt="You are a helpful mathematics tutor who explains concepts clearly."
    )

    response = await agent_with_system.invoke_async("Explain what derivatives are in calculus")
    print("Math tutor response:", response)

if __name__ == "__main__":
    asyncio.run(main())