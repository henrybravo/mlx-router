import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments, kernel_function
from openai import AsyncOpenAI

# Define a simulated external function to get weather forecast
class WeatherPlugin:
    @kernel_function(
        name="getForecast",
        description="Returns a simulated weather forecast for a given city."
    )
    async def get_forecast(self, city: str) -> str:
        # Simulate a weather forecast response
        return f"Sunny with a high of 25°C in {city}."

# Define the service ID
service_id = "openai_chat"

# Create a kernel
kernel = Kernel()

# Configure the AsyncOpenAI client for the local model
mlx_server = "http://localhost:8888/v1"
#mlx_model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
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

# Register the WeatherPlugin
kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

# Configure prompt execution settings
settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
settings.max_tokens = 2000
settings.temperature = 0.7
settings.top_p = 0.8

# Define the prompt template with variable and function call
prompt = """
I'm visiting {{$city}}. What are some activities I should do today? 
The weather today in {{$city}} is {{weather.getForecast $city}}.
"""

# Create the prompt template configuration
prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="city_activities",
    description="Suggests activities for visiting a city based on the weather.",
    execution_settings=settings
)

# Create a semantic function from the prompt
activities_function = kernel.add_function(
    function_name="suggest_activities",
    plugin_name="TravelPlanner",
    prompt_template_config=prompt_template_config
)

# Async function to invoke the prompt
async def run_prompt(city: str):
    try:
        # Create KernelArguments with the city variable
        arguments = KernelArguments(city=city)

        # Invoke on the KernelFunction object
        print(f"\n=== Invoking via KernelFunction for {city} ===")
        result = await activities_function.invoke(kernel, arguments)
        print(f"Output: {str(result)}")

        # Invoke on the kernel object
        print(f"\n=== Invoking via Kernel for {city} ===")
        result = await kernel.invoke(activities_function, arguments)
        print(f"Output: {str(result)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the async code
async def main():
    # Test with two cities
    await run_prompt(city="Rome")
    await run_prompt(city="Barcelona")

# Run the script
if __name__ == "__main__":
    asyncio.run(main())