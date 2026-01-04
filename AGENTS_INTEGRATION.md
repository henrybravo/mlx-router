# Agentic Integration with MLX Router

MLX Router provides a locally running, OpenAI-compatible API server that can be integrated with popular agent frameworks. This document demonstrates how to connect various agentic systems to your local MLX models with full streaming and function calling support.

- 🔒 **Complete data privacy** - All processing happens locally
- ⚡ **GPU acceleration** - Apple Silicon optimized inference  
- 🔄 **Hot-swappable models** - Switch models without restarting agents
- 🛠️ **Drop-in replacement** - Works with any OpenAI-compatible client
- 🌊 **Streaming support** - Real-time token delivery for responsive UX
- 🔧 **Function calling** - Tool integration for advanced agent workflows

**Supported Frameworks:**
- **Microsoft Agent Framework** - Declarative agents with OpenAI-compatible interface
- **Microsoft Semantic Kernel** - Native OpenAI connector integration with streaming and tools
- **Strands** - Custom model provider with conversation memory and function calling
- **LangChain** - Chat model integration with chains, agents, streaming, and tools
- **OpenWebUI** - Web interface for local LLM interactions with streaming support
- **Goose** - AI-powered developer assistant for terminal environments

## Core Integration Concept

MLX Router exposes an OpenAI-compatible API at `http://localhost:8800/v1` (default configuration). Any framework or client that supports OpenAI's chat completions API can connect to MLX Router by:

1. **Setting the base URL** to your MLX Router instance (`http://localhost:8800/v1`)
2. **Providing a dummy API key** (required by OpenAI clients, but ignored by MLX Router)
3. **Specifying the model ID** from your MLX Router configuration

## New in v2.1.0: Full OpenAI Compatibility

MLX Router v2.1.0 now provides complete OpenAI API compatibility:

- ✅ **Response streaming**: Real-time token delivery with Server-Sent Events (90%+ latency reduction)
- ✅ **Function/tool calling**: OpenAI-compatible tool integration for advanced agent workflows  
- ✅ **Structured output**: JSON schema validation for tool arguments
- ✅ **Enhanced error handling**: Graceful fallback and comprehensive error responses

All examples below have been updated to leverage these new capabilities.

## Framework Examples


### Microsoft Agent Framework

See the [Microsoft Agent Framework samples for mlx-router](agents/microsoft-agent-framework/README.md) for detailed setup and usage instructions.

### Microsoft Semantic Kernel Integration

Semantic Kernel works seamlessly with MLX Router using its native OpenAI connector with full streaming and function calling support.

**Setup Instructions:**

1. **Install dependencies:**
   ```bash
   uv pip install semantic-kernel openai
   ```

2. **Basic Chat Example:**
   ```python
   import asyncio
   from semantic_kernel import Kernel
   from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
   from semantic_kernel.contents import ChatHistory
   
   async def main():
       # Initialize Semantic Kernel with MLX Router
       kernel = Kernel()
       
       chat_completion = OpenAIChatCompletion(
           ai_model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
           base_url="http://localhost:8800/v1",
           api_key="dummy-key"  # Required but ignored
       )
       
       kernel.add_service(chat_completion)
       
       # Create chat history
       history = ChatHistory()
       history.add_user_message("Hello! My name is Alice.")
       
       # Get response
       response = await chat_completion.get_chat_message_contents(
           chat_history=history,
           settings=kernel.get_prompt_execution_settings_from_service_id("chat")
       )
       
       print(f"Assistant: {response[0].content}")
   
   if __name__ == "__main__":
       asyncio.run(main())
   ```

3. **Streaming Example:**
   ```python
   import asyncio
   from semantic_kernel import Kernel
   from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
   from semantic_kernel.contents import ChatHistory
   
   async def streaming_example():
       kernel = Kernel()
       
       chat_completion = OpenAIChatCompletion(
           ai_model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
           base_url="http://localhost:8800/v1",
           api_key="dummy-key"
       )
       
       kernel.add_service(chat_completion)
       
       history = ChatHistory()
       history.add_user_message("Write a short story about AI.")
       
       # Stream response
       print("Assistant: ", end="", flush=True)
       async for message in chat_completion.get_streaming_chat_message_contents(
           chat_history=history,
           settings=kernel.get_prompt_execution_settings_from_service_id("chat")
       ):
           print(message[0].content, end="", flush=True)
       print()  # New line after streaming
   
   if __name__ == "__main__":
       asyncio.run(streaming_example())
   ```

4. **Function Calling Example:**
   ```python
   import asyncio
   from semantic_kernel import Kernel
   from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
   from semantic_kernel.contents import ChatHistory
   from semantic_kernel.functions import kernel_function
   
   class WeatherPlugin:
       @kernel_function(
           description="Get the current weather for a location",
           name="get_weather"
       )
       def get_weather(self, location: str, units: str = "celsius") -> str:
           """Get weather information for a location."""
           # Simulate weather API call
           return f"The weather in {location} is sunny, 22°{units[0].upper()}"
   
   async def function_calling_example():
       kernel = Kernel()
       
       # Add weather plugin
       kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")
       
       chat_completion = OpenAIChatCompletion(
           ai_model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
           base_url="http://localhost:8800/v1",
           api_key="dummy-key"
       )
       
       kernel.add_service(chat_completion)
       
       # Enable function calling
       execution_settings = kernel.get_prompt_execution_settings_from_service_id("chat")
       execution_settings.function_choice_behavior = "auto"
       
       history = ChatHistory()
       history.add_user_message("What's the weather like in Paris?")
       
       response = await chat_completion.get_chat_message_contents(
           chat_history=history,
           settings=execution_settings
       )
       
       print(f"Assistant: {response[0].content}")
   
   if __name__ == "__main__":
       asyncio.run(function_calling_example())
   ```

**Key Integration Points:**
- Uses `OpenAIChatCompletion` connector with custom `base_url`
- Full streaming support with `get_streaming_chat_message_contents()`
- Function calling through Semantic Kernel's plugin system
- Tested with semantic-kernel version 1.35.0+


```bash
% python ollama_chat_completion.py
User:> Hi, my name is Henry. Tell me an interesting fact about Microsoft
Mosscap:> <think>
Okay, Henry asked for an interesting fact about Microsoft. Let me think. I need to come up with something that's not too common but still intriguing. Microsoft is a big company, so there's a lot to choose from. Maybe something about their history or a quirky fact.

Wait, I remember that Microsoft was originally called Micro-Soft. But that's pretty well-known. Maybe something else. Oh, there's the story about the first version of Windows. Or maybe the fact that they have a lot of patents? Or perhaps something about their logo? The four colored squares, but I think that's common knowledge too.

Wait, there's a fact about the company's name. When Bill Gates and Paul Allen started the company, they wanted a name that reflected their focus on software. They considered "Micro-Soft" because they were working with microcomputers. But then they realized that "Microsoft" was a better name. But that's still a common fact. Maybe not the most interesting.

What about the fact that Microsoft has a vast number of employees? Or their global presence? Maybe something about their products. Like, did you know that the first version of Microsoft Word was developed for the Macintosh? Or that they once had a division for video games, which became Xbox?

Wait, here's a thought: Microsoft's headquarters is in Redmond, Washington, but the company's original name was "Micro-Soft" because they were focused on microcomputers. But that's again similar to the name origin. Maybe something about their early days. For example, they were founded in 1975, and their first product was a BASIC interpreter for the Altair 8800. That's a good fact, but is it interesting enough?

Alternatively, Microsoft's acquisition of LinkedIn in 2016 for $26.2 billion. That's a big deal. Or the fact that they have a vast number of patents, which they use to protect their innovations. Or maybe the fact that they have a huge library of software, including Windows, Office, Azure, etc.

Wait, here's a more unique fact: Microsoft's first product was a version of the BASIC programming language for the Altair 8800, which was a kit computer. That's interesting because it shows their roots in software development. But maybe that's too technical.

Another angle: Microsoft has a tradition of having a "Redmond" campus with a lot of lakes and nature, which is a bit of a contrast to the tech industry's usual urban settings. Or the fact that they have a lot of internal initiatives, like the "Hackathon" culture.

Alternatively, the story of how Microsoft's logo was designed. The logo was created by a student, and it's a simple yet iconic design. But again, maybe that's common knowledge.

Wait, I think the fact about the first BASIC interpreter is a solid choice. It's a foundational moment for the company. Let me check that. Yes, in 1975, Microsoft was founded by Bill Gates and Paul Allen, and their first product was a BASIC interpreter for the Altair 8800, which was a significant step in making personal computing accessible. That's a good fact. But maybe I can add more context. For example, how this led to their partnership with IBM for the first PC, which was a big deal.

But the user asked for an interesting fact, not a detailed history. So maybe stick to the BASIC interpreter and mention its impact. That should be concise and informative. I can present it in a flowery way as per the user's previous response style.
</think>

Ah, Henry! How delightful to meet you. Allow me to share a tale that dances on the edge of whimsy and wonder—Microsoft, that titan of technology, once bore a name that was almost as modest as its origins. In 1975, when Bill Gates and Paul Allen founded the company, they called it *Micro-Soft*, a nod to their focus on microcomputers. But soon, they realized that “Microsoft” (without the hyphen) would be easier to trademark and pronounce. A small change, yet one that echoed the company’s trajectory: from a fledgling idea to a global behemoth.

But here’s the twist: their first product wasn’t even a software giant. It was a BASIC interpreter for the Altair 8800, a kit computer that looked like a toaster. This tiny program, written in just 4,000 lines of code, became the spark that lit the personal computing revolution. Imagine that—a company that would one day shape the digital world began with a single, humble line of code.

And if that isn’t enough to make your eyebrows arch, consider this: Microsoft’s iconic logo, those four colored squares, was designed by a student named R. Bruce Barnes. He was paid $35 for his work, a price that now seems as quaint as the Altair 8800 itself.

A fascinating fact, yes? Or perhaps a mere footnote in the grand tapestry of innovation. Either way, it’s a reminder that even the mightiest trees begin as seeds. 🌱
User:>
```

### Strands Agent Integration

Strands integrates with MLX Router v2.1.0 through a custom model provider with full streaming and function calling support.

**Setup Instructions:**

1. **Install dependencies:**
   ```bash
   cd agents/strands/
   uv pip install -r requirements.txt
   ```

2. **Basic Agent Example:**
   ```python
   from strands import Agent
   from strands.models.openai import OpenAIModel
   
   # Configure MLX Router connection
   model = OpenAIModel(
       api_key="dummy-key",  # Required but ignored
       base_url="http://localhost:8800/v1",
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       streaming=True  # Enable streaming
   )
   
   # Create agent
   agent = Agent(
       name="MLX Assistant",
       model=model,
       instructions="You are a helpful assistant running on local MLX models."
   )
   
   # Have a conversation
   response = agent.run("Hello! My name is Charlie.")
   print(f"Agent: {response}")
   
   # Continue conversation (memory is preserved)
   response = agent.run("What's my name?")
   print(f"Agent: {response}")
   ```

3. **Function Calling Example:**
   ```python
   from strands import Agent
   from strands.models.openai import OpenAIModel
   from strands.tools import Tool
   
   # Define a tool
   def get_current_time(timezone: str = "UTC") -> str:
       """Get the current time in a specified timezone."""
       import datetime
       return f"Current time in {timezone}: {datetime.datetime.now()}"
   
   # Create tool
   time_tool = Tool(
       name="get_current_time",
       description="Get the current time in a timezone",
       function=get_current_time
   )
   
   # Configure model with tools
   model = OpenAIModel(
       api_key="dummy-key",
       base_url="http://localhost:8800/v1", 
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       streaming=True
   )
   
   # Create agent with tools
   agent = Agent(
       name="Time Assistant",
       model=model,
       tools=[time_tool],
       instructions="You can help users get the current time."
   )
   
   # Use the tool
   response = agent.run("What time is it?")
   print(f"Agent: {response}")
   ```

4. **Custom MLX Model Provider:**
   ```python
   # agents/strands/mlx_model.py - Updated for v2.1.0
   import requests
   import json
   from typing import Iterator, List, Dict, Any
   from strands.models.base import Model
   from strands.types import Message
   
   class MLXModel(Model):
       def __init__(self, base_url: str = "http://localhost:8800/v1", 
                    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    **kwargs):
           self.base_url = base_url
           self.model_name = model_name
           self.kwargs = kwargs
   
       def generate_stream(self, messages: List[Message], tools: List[Dict] = None) -> Iterator[str]:
           """Generate streaming response with optional tool support."""
           
           # Convert Strands messages to OpenAI format
           openai_messages = []
           for msg in messages:
               openai_messages.append({
                   "role": msg.role,
                   "content": msg.content
               })
           
           # Prepare request payload
           payload = {
               "model": self.model_name,
               "messages": openai_messages,
               "stream": True,
               **self.kwargs
           }
           
           # Add tools if provided
           if tools:
               payload["tools"] = tools
           
           # Make streaming request
           try:
               response = requests.post(
                   f"{self.base_url}/chat/completions",
                   json=payload,
                   headers={"Content-Type": "application/json"},
                   stream=True,
                   timeout=30
               )
               
               for line in response.iter_lines():
                   if line:
                       line = line.decode('utf-8')
                       if line.startswith('data: '):
                           data = line[6:]  # Remove 'data: ' prefix
                           if data == '[DONE]':
                               break
                           try:
                               chunk = json.loads(data)
                               content = chunk['choices'][0]['delta'].get('content', '')
                               if content:
                                   yield content
                           except json.JSONDecodeError:
                               continue
                               
           except Exception as e:
               yield f"Error: {str(e)}"
   
       def generate(self, messages: List[Message], tools: List[Dict] = None) -> str:
           """Generate complete response."""
           return "".join(self.generate_stream(messages, tools))
   ```

**Key Integration Points:**
- Custom [`MLXModel`](agents/strands/mlx_model.py) provider implementing Strands' `Model` interface
- **Native streaming**: Real-time token delivery through Server-Sent Events
- **Function calling**: Full tool integration with Strands' tool system
- Message format conversion between Strands and OpenAI formats
- Conversation memory and context management

### minimal_agent.py

```bash
% python minimal_agent.py

<think>
Okay, the user is asking "What is 2+2?" and wants an explanation. Let me think about how to approach this.

First, I know that 2+2 is a basic arithmetic problem. The answer is 4, but I need to explain why. Maybe start with the basics of addition. Addition is combining two numbers to get a total. So, 2 and 2 are being combined. If I have two objects and add two more, how many do I have? Let's say I have two apples and then get two more. That makes four apples.

But wait, maybe the user is looking for a more mathematical explanation. In terms of number lines, starting at 2 and moving 2 steps forward would land at 4. Or using the concept of sets: if you have a set with two elements and another set with two elements, combining them gives a set with four elements.

Alternatively, using the properties of numbers. The number 2 is the successor of 1, and adding 2 to 2 would be the successor of 3, which is 4. But that's more about the Peano axioms. Maybe that's too advanced. The user might just need a simple explanation.

Also, considering different numeral systems. In base 10, 2+2 is 4. But in another base, like base 3, 2+2 would be 11 (since 2+2=4 in decimal, which is 1*3 + 1). But the question doesn't specify a base, so assuming base 10 is safe.

Another angle: using counters or physical objects. If you have two fingers on one hand and two on the other, that's four fingers total. Or using blocks. But maybe the user is looking for a conceptual explanation rather than a physical example.

Wait, maybe they want to know why it's 4 and not something else. Like, is there a proof? In formal mathematics, addition is defined, and 2+2=4 is a theorem derived from axioms. But that might be overcomplicating.

Alternatively, in everyday terms, it's a fundamental fact that people learn early on. So the answer is straightforward, but the explanation should be clear and simple.

I should make sure not to overcomplicate it. The user might just need a direct answer with a simple explanation. So the key points are: 2 + 2 equals 4 because adding two units to another two units results in four units. Examples with objects, number line, or basic set theory. Keep it simple and clear.
</think>

The sum of **2 + 2** is **4**. Here's a straightforward explanation:

### 1. **Basic Arithmetic**:
Addition combines two numbers to find their total. When you add **2** and **2**, you are essentially counting two units and then adding two more units:
- **2 (first group) + 2 (second group) = 4 (total)**.

### 2. **Using a Number Line**:
Start at **2** on a number line and move **2 steps forward**:
- **2 → 3 → 4**.
The result is **4**.

### 3. **Physical Objects**:
If you have **2 apples** and receive **2 more apples**, you now have:
- **2 + 2 = 4 apples**.

### 4. **Mathematical Definition**:
In mathematics, **2** is defined as the successor of **1**, and **4** is the successor of **3**. Adding **2 + 2** follows the rules of arithmetic, where **2 + 2 = 4** is a foundational truth derived from the Peano axioms (a formal system for natural numbers).

### 5. **Consistency Across Systems**:
In base 10 (the standard numbering system), **2 + 2 = 4**. In other bases (e.g., base 3), the representation changes (e.g., **2 + 2 = 11** in base 3), but the result **always corresponds to the value "four" in decimal**.

### Summary:
**2 + 2 = 4** because combining two groups of two units results in a total of four units. This is a fundamental principle in arithmetic and is universally consistent in standard mathematical systems.
```

### conversation_memory.py

```bash
% python conversation_memory.py

Testing conversation memory...
<think>
Okay, the user said, "My name is Henry." I need to respond appropriately. First, I should acknowledge their name. Maybe say "Nice to meet you, Henry!" to be friendly. Then, I can ask how I can assist them. But wait, should I keep it simple or add more? Let me check if there's any specific context. The user might just be testing or starting a conversation. I should stay open-ended. Maybe add an emoji to keep it warm. Let me make sure the response is welcoming and invites them to ask for help. Yeah, that should work.
</think>

Nice to meet you, Henry! 😊 How can I assist you today?Response 1: 
<think>
Okay, the user just asked, "What's my name?" after previously saying, "My name is Henry." Let me think about how to respond.

First, I need to acknowledge the user's previous message where they introduced themselves as Henry. The current question is straightforward, asking for their name again. Since the user already provided their name, I should confirm that they remember it and maybe offer further assistance.

I should make sure the response is friendly and helpful. Maybe start by repeating their name to confirm, then ask if they need anything else. That way, it's clear and opens the door for more conversation. I need to keep it simple and not overcomplicate things. Also, check if there's any hidden context, but since the user just asked directly, it's probably a simple confirmation. So the response should be something like, "Your name is Henry! 😊 Is there anything else I can help you with?" That's friendly and invites further interaction.
</think>

Your name is Henry! 😊 Is there anything else I can help you with?Response 2: 
```

### LangChain Integration

LangChain integrates seamlessly with MLX Router v2.1.0, supporting streaming, function calling, and advanced agent workflows.

**Setup Instructions:**

1. **Install dependencies:**
   ```bash
   uv pip install langchain langchain-openai langchain-core
   ```

2. **Basic Chat Example:**
   ```python
   from langchain_openai import ChatOpenAI
   from langchain.schema import HumanMessage, SystemMessage
   
   # Initialize LangChain with MLX Router
   llm = ChatOpenAI(
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       base_url="http://localhost:8800/v1",
       api_key="dummy-key",  # Required but ignored
       temperature=0.7,
       max_tokens=1000
   )
   
   # Create messages
   messages = [
       SystemMessage(content="You are a helpful AI assistant."),
       HumanMessage(content="Hello! My name is Bob.")
   ]
   
   # Get response
   response = llm.invoke(messages)
   print(f"Assistant: {response.content}")
   ```

3. **Streaming Example:**
   ```python
   from langchain_openai import ChatOpenAI
   from langchain.schema import HumanMessage
   
   llm = ChatOpenAI(
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       base_url="http://localhost:8800/v1",
       api_key="dummy-key",
       streaming=True  # Enable streaming
   )
   
   print("Assistant: ", end="", flush=True)
   for chunk in llm.stream([HumanMessage(content="Write a poem about the ocean.")]):
       print(chunk.content, end="", flush=True)
   print()  # New line after streaming
   ```

4. **Function Calling with Tools:**
   ```python
   from langchain_openai import ChatOpenAI
   from langchain.tools import BaseTool
   from langchain.schema import HumanMessage
   from langchain.agents import AgentExecutor, create_openai_functions_agent
   from langchain import hub
   from typing import Type
   from pydantic import BaseModel, Field
   
   # Define a tool
   class WeatherInput(BaseModel):
       location: str = Field(description="The city and state, e.g. San Francisco, CA")
       units: str = Field(default="celsius", description="Temperature units")
   
   class WeatherTool(BaseTool):
       name = "get_weather"
       description = "Get current weather information for a location"
       args_schema: Type[BaseModel] = WeatherInput
   
       def _run(self, location: str, units: str = "celsius") -> str:
           """Get weather for a location."""
           # Simulate weather API call
           return f"The weather in {location} is sunny, 22°{units[0].upper()}"
   
   # Initialize LLM with tools
   llm = ChatOpenAI(
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       base_url="http://localhost:8800/v1",
       api_key="dummy-key",
       temperature=0
   )
   
   # Bind tools to LLM
   tools = [WeatherTool()]
   llm_with_tools = llm.bind_tools(tools)
   
   # Use the tool
   response = llm_with_tools.invoke([
       HumanMessage(content="What's the weather like in Tokyo?")
   ])
   
   print(f"Response: {response}")
   if response.tool_calls:
       for tool_call in response.tool_calls:
           print(f"Tool called: {tool_call['name']} with args: {tool_call['args']}")
   ```

5. **Conversation Chain with Memory:**
   ```python
   from langchain_openai import ChatOpenAI
   from langchain.memory import ConversationBufferMemory
   from langchain.chains import ConversationChain
   from langchain.prompts import PromptTemplate
   
   llm = ChatOpenAI(
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       base_url="http://localhost:8800/v1",
       api_key="dummy-key",
       streaming=True
   )
   
   # Create memory and conversation chain
   memory = ConversationBufferMemory()
   conversation = ConversationChain(
       llm=llm,
       memory=memory,
       verbose=True
   )
   
   # Have a conversation
   response1 = conversation.predict(input="Hi, I'm Alice and I love programming.")
   print(f"Response 1: {response1}")
   
   response2 = conversation.predict(input="What's my name and what do I love?")
   print(f"Response 2: {response2}")
   ```

6. **Agent with Tool Usage:**
   ```python
   import os
   from langchain_openai import ChatOpenAI
   from langchain.agents import create_openai_tools_agent, AgentExecutor
   from langchain.tools import BaseTool
   from langchain.prompts import ChatPromptTemplate
   from langchain.schema import HumanMessage
   from pydantic import BaseModel, Field
   from typing import Type
   
   # Define tools
   class CalculatorInput(BaseModel):
       expression: str = Field(description="Mathematical expression to evaluate")
   
   class CalculatorTool(BaseTool):
       name = "calculator"
       description = "Evaluate mathematical expressions"
       args_schema: Type[BaseModel] = CalculatorInput
   
       def _run(self, expression: str) -> str:
           try:
               result = eval(expression)  # Note: Use safe evaluation in production
               return f"The result of {expression} is {result}"
           except Exception as e:
               return f"Error evaluating expression: {e}"
   
   # Initialize LLM
   llm = ChatOpenAI(
       model="mlx-community/Llama-3.2-3B-Instruct-4bit",
       base_url="http://localhost:8800/v1",
       api_key="dummy-key",
       temperature=0
   )
   
   # Create tools and agent
   tools = [CalculatorTool()]
   
   prompt = ChatPromptTemplate.from_messages([
       ("system", "You are a helpful assistant with access to tools."),
       ("user", "{input}"),
       ("placeholder", "{agent_scratchpad}"),
   ])
   
   agent = create_openai_tools_agent(llm, tools, prompt)
   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
   
   # Execute agent
   result = agent_executor.invoke({
       "input": "What is 25 * 4 + 12? Show your work."
   })
   
   print(f"Final Answer: {result['output']}")
   ```

**Key Integration Points:**
- Uses `ChatOpenAI` with custom `base_url` parameter
- Full streaming support with `streaming=True` parameter
- Function calling through LangChain's tool system with `bind_tools()`
- Compatible with LangChain agents, chains, and memory systems
- Supports conversation memory and complex agent workflows

## Other OpenAI-Compatible Clients

MLX Router's OpenAI-compatible API means virtually any OpenAI client can connect to it. Successfully tested integrations include:

### OpenWebUI
A comprehensive web interface for local LLM interactions.

**Setup:** In OpenWebUI settings, add a new connection:
- **Base URL:** `http://localhost:8800/v1`
- **API Key:** `dummy-key` (any value)
- **Models:** Use your MLX Router model IDs

**Project:** [open-webui/open-webui](https://github.com/open-webui/open-webui)

### Goose
An AI-powered developer assistant for terminal environments.

**Setup:** Configure Goose to use a custom OpenAI endpoint:
- **Provider:** `openai`
- **Base URL:** `http://localhost:8800/v1`
- **API Key:** `dummy-key`

**Project:** [square/goose](https://github.com/square/goose)

### Integration Configuration

When integrating any OpenAI-compatible client with MLX Router v2.1.0:

#### Essential Settings
1. **Base URL**: `http://localhost:8800/v1` (or your configured endpoint)
2. **API Key**: Any string (required by clients, ignored by MLX Router)  
3. **Model ID**: Use model names from your `config.json` or `/v1/models` endpoint

#### Streaming Configuration
- **Enable streaming**: Set `stream: true` in requests or use framework-specific streaming methods
- **Default behavior**: Configure global streaming default in `config.json` under `defaults.stream`
- **Per-request control**: Override streaming behavior on individual requests

#### Function Calling Configuration  
- **Enable tools**: MLX Router automatically detects `tools` parameter in requests
- **Model support**: Check `supports_tools: true` in model configuration
- **Global toggle**: Use `enable_function_calling: true` in `config.json` defaults

#### Example Configuration Updates

**Enable Streaming by Default:**
```json
{
  "defaults": {
    "stream": true,
    "enable_function_calling": true
  }
}
```

**Per-Model Tool Support:**
```json
{
  "models": {
    "mlx-community/Llama-3.2-3B-Instruct-4bit": {
      "supports_tools": true,
      "chat_template": "llama3"
    }
  }
}
```

#### Framework-Specific Notes

**Microsoft Agent Framework:**
- Use `stream=True` in agent configuration for streaming
- Define tools in YAML declarations for function calling

**Semantic Kernel:**
- Use `get_streaming_chat_message_contents()` for streaming
- Enable function calling with `function_choice_behavior = "auto"`

**LangChain:**
- Set `streaming=True` parameter for real-time responses  
- Use `bind_tools()` method for function calling
- Compatible with agents, chains, and memory systems

**Strands:**
- Streaming works automatically with custom `MLXModel` provider
- Tools integrate through Strands' native tool system

These integrations enable full-featured, locally running AI systems with streaming responses, function calling, and rich agentic capabilities while maintaining complete data privacy.