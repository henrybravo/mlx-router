from agents.strands.minimal_agent import MLXModel
from agents.strands.mlx_model import Agent

# Initialize your custom MLX model
mlx_model = MLXModel(
    base_url="http://localhost:8800/v1",
    model_id="mlx-community/Qwen3-30B-A3B-8bit",
    api_key="strands-key",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

# Create agent with your custom model
agent = Agent(model=mlx_model)

# Test it
response = await agent.invoke_async("What is 2+2? Please explain your reasoning.")
print(response)
