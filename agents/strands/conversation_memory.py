from mlx_model import MLXModel
from strands import Agent

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

# Test conversation memory - multiple turns
print("Testing conversation memory...")
response1 = await agent.invoke_async("My name is Henry")
print("Response 1:", response1)

response2 = await agent.invoke_async("What's my name?")  # Should remember
print("Response 2:", response2)