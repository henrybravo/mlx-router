# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os
from pathlib import Path
from random import randint
from typing import Literal

from dotenv import load_dotenv
from agent_framework.declarative import AgentFactory


def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """A simple function tool to get weather information."""
    return f"The weather in {location} is {randint(-10, 30) if unit == 'celsius' else randint(30, 100)} degrees {unit}."


async def main():
    """Create an agent from a declarative yaml specification and run it."""
    load_dotenv()
    router_base_url = os.environ.get("MLX_ROUTER_BASE_URL", "http://localhost:8800/v1")
    os.environ.setdefault("OPENAI_BASE_URL", router_base_url)
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("MLX_ROUTER_API_KEY", "dummy-key"))
    os.environ.setdefault("OPENAI_MODEL", os.environ.get("MLX_ROUTER_MODEL", "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"))

    current_path = Path(__file__).parent
    # Move two levels up to reach agents/microsoft-agent-framework
    yaml_path = current_path.parent / "agent-declarations" / "chatclient_GetWeather.yaml"

    # load the yaml from the path
    with yaml_path.open("r") as f:
        yaml_str = f.read()

    # Inline env expansion because Agent Framework does not expand =Env placeholders
    yaml_str = yaml_str.replace("=Env.OPENAI_MODEL", os.environ["OPENAI_MODEL"]).replace("=Env.OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])

    agent_factory = AgentFactory(bindings={"get_weather": get_weather})
    agent = agent_factory.create_agent_from_yaml(yaml_str)
    # use the agent
    response = await agent.run("What's the weather in Amsterdam, in celsius?")
    print("Agent response:", response.text)


if __name__ == "__main__":
    asyncio.run(main())
