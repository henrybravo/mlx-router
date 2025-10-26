# This script requires mlx_model.py to be in the same directory
# mlx_model.py provides the custom MLXModel class for Strands integration with MLX Router
from mlx_model import MLXModel
import asyncio
from mlx_model import MLXModel
from strands import Agent

async def main():
    # Initialize your custom MLX model
    mlx_model = MLXModel(
        base_url="http://localhost:8800/v1",
        model_id="mlx-community/gpt-oss-120b-MXFP4-Q8",
        api_key="strands-key",
        params={
            "max_tokens": 16384,
            "temperature": 0.7,
        }
    )

    # Create agent with your custom model
    agent = Agent(model=mlx_model)

    # Test it
    response = await agent.invoke_async("What is 2+2? Please explain your reasoning.")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
