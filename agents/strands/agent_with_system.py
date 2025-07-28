from mlx_model import MLXModel
from strands import Agent

# Initialize your custom MLX model
mlx_model = MLXModel(
    base_url="http://host.docker.internal:8888/v1",
    model_id="mlx-community/Qwen3-30B-A3B-8bit",
    api_key="strands-key",
    params={
        "max_tokens": 5000,
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