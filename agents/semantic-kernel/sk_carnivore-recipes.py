import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from openai import AsyncOpenAI

import argparse

parser = argparse.ArgumentParser(description="Generate carnivore-friendly breakfast recipes")
parser.add_argument("--num_recipes", default="3", help="Number of recipes to generate")
parser.add_argument("--meat_type", default="beef", help="Type of meat to use")
args = parser.parse_args()

# Define the system message (prompt)
system_message = """
I'm a carnivore in search of new recipes. I love grilling!
Can you give me a list of {{ $num_recipes }} breakfast recipes that are carnivore-friendly using {{ $meat_type }}?
"""

mlx_server="http://localhost:8800/v1"
# mlx_model="mlx-community/Llama-3.2-3B-Instruct-4bit"
# mlx_model="deepseek-ai/deepseek-coder-6.7b-instruct"
# mlx_model="mlx-community/Phi-4-reasoning-plus-6bit"
#mlx_model="mlx-community/Qwen3-30B-A3B-8bit"
mlx_model = "mlx-community/gpt-oss-120b-MXFP4-Q8"

# Define the service ID
service_id = "openai_chat"

# Create a kernel
kernel = Kernel()

# Configure the AsyncOpenAI client for the local model
openAIClient = AsyncOpenAI(
    api_key="sk-key",  # Use a placeholder API key for local models
    base_url=mlx_server,  # Local endpoint for your model server
)

# Add the OpenAI chat completion service to the kernel
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id=mlx_model,
        async_client=openAIClient
    )
)

# Configure prompt execution settings
settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
settings.max_tokens = 2000
settings.temperature = 0.7
settings.top_p = 0.8

# Define the prompt template configuration
prompt_template_config = PromptTemplateConfig(
    template=system_message,
    name="carnivore_recipes",
    description="Generates a list of carnivore-friendly breakfast recipes.",
    execution_settings=settings
)

# Create a semantic function from the prompt
recipe_function = kernel.add_function(
    function_name="generate_recipes",
    plugin_name="CarnivoreRecipes",
    prompt_template_config=prompt_template_config
)

# Async function to invoke the kernel and get the response
async def get_recipes(num_recipes, meat_type):
    try:
        result = await kernel.invoke(
            recipe_function,
            num_recipes=num_recipes,
            meat_type=meat_type
        )
        output = str(result)
        print("Carnivore-Friendly Breakfast Recipes:")
        print(output)
        with open("recipes.md", "a") as f:
            f.write(output)
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the async code
async def main():
    await get_recipes(num_recipes=args.num_recipes, meat_type=args.meat_type)

# Run the script
if __name__ == "__main__":
    asyncio.run(main())