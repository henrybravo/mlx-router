# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from agent_framework.declarative import AgentFactory


async def main():
    """Create an agent from a declarative yaml specification and run it."""
    load_dotenv()
    # Configure mlx-router as OpenAI-compatible endpoint
    router_base_url = os.environ.get("MLX_ROUTER_BASE_URL", "http://localhost:8800/v1")
    os.environ.setdefault("OPENAI_BASE_URL", router_base_url)
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("MLX_ROUTER_API_KEY", "dummy-key"))
    os.environ.setdefault("OPENAI_MODEL", os.environ.get("MLX_ROUTER_MODEL", "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"))

    # get the path (two levels up to microsoft-agent-framework)
    current_path = Path(__file__).parent
    yaml_path = current_path.parent / "agent-declarations" / "OpenAIResponses.yaml"

    # load the yaml from the path
    with yaml_path.open("r") as f:
        yaml_str = f.read()

    yaml_str = yaml_str.replace("=Env.OPENAI_MODEL", os.environ["OPENAI_MODEL"]).replace("=Env.OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]).replace("=Env.OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))

    # create the agent from the yaml
    agent = AgentFactory().create_agent_from_yaml(yaml_str)
    # use the agent
    response = await agent.run("Why is the sky blue, answer in Dutch?")
    print("Agent response:", response.value)


if __name__ == "__main__":
    asyncio.run(main())
