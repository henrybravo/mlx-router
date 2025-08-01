import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from openai import AsyncOpenAI

# Define the service ID
service_id = "openai_chat"

# Create a kernel
kernel = Kernel()

# Configure the AsyncOpenAI client for the local model
mlx_server = "http://localhost:8888/v1"
mlx_model = "mlx-community/Llama-3.3-70B-Instruct-6bit"
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
settings.max_tokens = 4096
settings.temperature = 0.4
settings.top_p = 0.8

# Zero-shot prompt: Determine intent without examples
zero_shot_prompt = """
Instructions: What is the intent of this request?
If you don't know the intent, don't guess; instead respond with "Unknown".
Choices: SendEmail, SendMessage, CompleteTask, CreateDocument, Unknown.
User Input: {{ $request }}
Intent:
"""
zero_shot_config = PromptTemplateConfig(
    template=zero_shot_prompt,
    name="zero_shot_intent",
    description="Determines the intent of a user request using zero-shot learning.",
    execution_settings=settings
)
zero_shot_function = kernel.add_function(
    function_name="zero_shot_intent",
    plugin_name="IntentDetection",
    prompt_template_config=zero_shot_config
)

# Few-shot prompt: Determine intent with examples
few_shot_prompt = """
Instructions: What is the intent of this request?
If you don't know the intent, don't guess; instead respond with "Unknown".
Choices: SendEmail, SendMessage, CompleteTask, CreateDocument, Unknown.

User Input: Can you send a very quick approval to the marketing team?
Intent: SendMessage

User Input: Can you send the full update to the marketing team?
Intent: SendEmail

User Input: {{ $request }}
Intent:
"""
few_shot_config = PromptTemplateConfig(
    template=few_shot_prompt,
    name="few_shot_intent",
    description="Determines the intent of a user request using few-shot learning.",
    execution_settings=settings
)
few_shot_function = kernel.add_function(
    function_name="few_shot_intent",
    plugin_name="IntentDetection",
    prompt_template_config=few_shot_config
)

# Persona prompt: Explain asynchronous programming as a software engineer
persona_prompt = """
You are a highly experienced software engineer. Explain the concept of asynchronous programming to a beginner in a clear and concise manner.
"""
persona_config = PromptTemplateConfig(
    template=persona_prompt,
    name="persona_explanation",
    description="Explains asynchronous programming as a software engineer.",
    execution_settings=settings
)
persona_function = kernel.add_function(
    function_name="persona_explanation",
    plugin_name="TechExplainer",
    prompt_template_config=persona_config
)

# Chain-of-thought prompt: Calculate apples eaten by the farmer
cot_prompt = """
Instructions: A farmer has 150 apples and wants to sell them in baskets. Each basket can hold 12 apples. If any apples remain after filling as many baskets as possible, the farmer will eat them. How many apples will the farmer eat?

First, calculate how many full baskets the farmer can make by dividing the total apples by the apples per basket:
1. 

Next, subtract the number of apples used in the baskets from the total number of apples to find the remainder: 
1.

Finally, the farmer will eat the remaining apples:
1.
"""
cot_config = PromptTemplateConfig(
    template=cot_prompt,
    name="chain_of_thought",
    description="Calculates remaining apples using chain-of-thought reasoning.",
    execution_settings=settings
)
cot_function = kernel.add_function(
    function_name="chain_of_thought",
    plugin_name="MathSolver",
    prompt_template_config=cot_config
)

# Async function to invoke all prompts
async def run_prompts():
    try:
        # Zero-shot example
        print("\n=== Zero-Shot Intent Detection ===")
        result = await kernel.invoke(
            zero_shot_function,
            request="Draft a proposal for the new project"
        )
        print(f"Input: Draft a proposal for the new project")
        print(f"Output: {str(result)}")

        # Few-shot example
        print("\n=== Few-Shot Intent Detection ===")
        result = await kernel.invoke(
            few_shot_function,
            request="Draft a proposal for the new project"
        )
        print(f"Input: Draft a proposal for the new project")
        print(f"Output: {str(result)}")

        # Persona example
        print("\n=== Persona-Based Explanation ===")
        result = await kernel.invoke(persona_function)
        print(f"Output: {str(result)}")

        # Chain-of-thought example
        print("\n=== Chain-of-Thought Reasoning ===")
        result = await kernel.invoke(cot_function)
        print(f"Output: {str(result)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the async code
async def main():
    await run_prompts()

# Run the script
if __name__ == "__main__":
    asyncio.run(main())