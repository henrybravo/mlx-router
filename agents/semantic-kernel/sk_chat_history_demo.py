import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from openai import AsyncOpenAI

# Define the service ID
service_id = "openai_chat"

# Create a kernel
kernel = Kernel()

# Configure the AsyncOpenAI client for the local model
mlx_server = "http://localhost:8800/v1"
# mlx_model = "mlx-community/Llama-3.3-70B-Instruct-8bit"
mlx_model = "mlx-community/gpt-oss-120b-MXFP4-Q8"
openAIClient = AsyncOpenAI(
    api_key="sk-key",  # Use a placeholder API key for local models
    base_url=mlx_server,  # Local endpoint for your model server
)

# Add the OpenAI chat completion service to the kernel
chat_completion = OpenAIChatCompletion(
    service_id=service_id,
    ai_model_id=mlx_model,
    async_client=openAIClient
)
kernel.add_service(chat_completion)

# Configure prompt execution settings
settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
settings.max_tokens = 2000
settings.temperature = 0.7
settings.top_p = 0.8

# Async function to run chat history examples
async def run_chat_history():
    try:
        # Example 1: Simple ChatHistory
        print("\n=== Simple ChatHistory Example ===")
        chat_history = ChatHistory()
        chat_history.add_system_message("You are a helpful assistant managing a restaurant order system.")
        chat_history.add_user_message("What's available to order?")
        chat_history.add_assistant_message("We have pizza, pasta, and salad available to order. What would you like to order?")
        chat_history.add_user_message("I'd like to have the first option, please.")

        # Print chat history
        for message in chat_history:
            role = str(message.role).lower()  # Ensure role is a string
            print(f"{role}: {message.content}")

        # Add the new user message for the response
        chat_history.add_user_message("Can you confirm my order for pizza?")

        # Get response from model
        response = await chat_completion.get_chat_message_contents(
            chat_history=chat_history,
            settings=settings
        )
        print(f"assistant: {response[0].content}")

        # Example 2: ChatHistory with ChatMessageContent
        print("\n=== ChatHistory with ChatMessageContent ===")
        chat_history = ChatHistory()
        chat_history.add_system_message("You are a helpful assistant managing a restaurant order system.")
        chat_history.add_message(
            message=ChatMessageContent(
                role="user",
                author_name="Laimonis Dumins",
                items=[TextContent(text="What available on this menu?")]
            )
        )
        chat_history.add_assistant_message("We have pizza, pasta, and salad available to order. What would you like to order?")

        # Print chat history
        for message in chat_history:
            role = str(message.role).lower()  # Ensure role is a string
            if hasattr(message, "author_name") and message.author_name:
                print(f"authorrole.{role} ({message.author_name}):")
            else:
                print(f"authorrole.{role}:")
            for item in message.items:
                if isinstance(item, TextContent):
                    print(f"  Text: {item.text}")

        # Add the new user message for the response
        chat_history.add_user_message("I'd like to order pasta, please.")

        # Get response from model
        response = await chat_completion.get_chat_message_contents(
            chat_history=chat_history,
            settings=settings
        )
        print(f"assistant: {response[0].content}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the async code
async def main():
    await run_chat_history()

# Run the script
if __name__ == "__main__":
    asyncio.run(main())