import asyncio
import yaml
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.handlebars_prompt_template import HandlebarsPromptTemplate
from semantic_kernel.functions import KernelArguments
from openai import AsyncOpenAI

# Define the inline Handlebars prompt template
handlebars_template = """
<message role="system">You are an AI assistant designed to help with image recognition tasks.</message>
<message role="user">
    <text>{{request}}</text>
    <imageDescription>{{imageDescription}}</imageDescription>
</message>
"""

# Define the service ID
service_id = "openai_chat"

# Create a kernel
kernel = Kernel()

# Configure the AsyncOpenAI client for the local model
mlx_server = "http://localhost:8888/v1"
mlx_model = "mlx-community/Llama-3.3-70B-Instruct-8bit"
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

# Create the inline prompt template configuration
inline_prompt_config = PromptTemplateConfig(
    template=handlebars_template,
    template_format="handlebars",
    name="Vision_Chat_Inline",
    description="Inline vision chat prompt template for text-based image descriptions.",
    execution_settings=settings
)

# Load YAML prompt from file
with open("HandlebarsPrompt.yaml", "r") as f:
    yaml_prompt = yaml.safe_load(f)

# Create the YAML prompt template configuration
yaml_prompt_config = PromptTemplateConfig(
    template=yaml_prompt["template"],
    template_format="handlebars",
    name=yaml_prompt.get("name", "Vision_Chat_Prompt"),
    description=yaml_prompt.get("description", "Vision chat prompt template for text-based image descriptions."),
    execution_settings=settings
)

# Async function to invoke the prompts
async def run_prompts():
    try:
        # Define arguments
        arguments = KernelArguments(
            request="Describe this image:",
            imageDescription="The image is a solid block of bright red color."
        )

        # Invoke inline prompt
        print("\n=== Inline Handlebars Prompt ===")
        result = await kernel.invoke_prompt(
            prompt=inline_prompt_config.template,
            template_format="handlebars",
            arguments=arguments,
            service_id=service_id
        )
        print(f"Output: {str(result)}")

        # Invoke YAML prompt
        print("\n=== YAML Handlebars Prompt ===")
        result = await kernel.invoke_prompt(
            prompt=yaml_prompt_config.template,
            template_format="handlebars",
            arguments=arguments,
            service_id=service_id
        )
        print(f"Output: {str(result)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the async code
async def main():
    await run_prompts()

# Run the script
if __name__ == "__main__":
    asyncio.run(main())
