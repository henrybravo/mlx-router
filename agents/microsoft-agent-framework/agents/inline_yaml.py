# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os

from dotenv import load_dotenv
from agent_framework.declarative import AgentFactory

"""
This sample shows how to create an agent using an inline YAML string rather than a file.

It uses the MLX router via OpenAI-compatible settings, configured through environment variables.

Prerequisites:
- `uv pip install agent-framework-declarative --pre`
- Set (or rely on defaults):
    - MLX_ROUTER_BASE_URL (default http://localhost:8800/v1)
    - MLX_ROUTER_API_KEY (default dummy-key)
    - MLX_ROUTER_MODEL (default mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit)
"""


async def main():
    """Create an agent from a declarative YAML specification and run it."""
    load_dotenv()
    router_base_url = os.environ.get("MLX_ROUTER_BASE_URL", "http://localhost:8800/v1")
    os.environ.setdefault("OPENAI_BASE_URL", router_base_url)
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("MLX_ROUTER_API_KEY", "dummy-key"))
    os.environ.setdefault("OPENAI_MODEL", os.environ.get("MLX_ROUTER_MODEL", "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"))

    yaml_definition = f"""kind: Prompt
name: DiagnosticAgent
displayName: Diagnostic Assistant
instructions: Specialized diagnostic and issue detection agent for systems with critical error protocol and automatic handoff capabilities
description: A agent that performs diagnostics on systems and can escalate issues when critical errors are detected.

model:
    id: {os.environ['OPENAI_MODEL']}
    provider: OpenAI
    apiType: Chat
    connection:
        kind: key
        key: {os.environ['OPENAI_API_KEY']}
"""
    async with AgentFactory().create_agent_from_yaml(yaml_definition) as agent:
        response = await agent.run("What can you do for me?")
        print("Agent response:", response.text)


if __name__ == "__main__":
    asyncio.run(main())
