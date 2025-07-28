# Agentic Integration with MLX Router

MLX Router provides a locally running, OpenAI-compatible API server that can be integrated with popular agent frameworks. This document demonstrates how to connect various agentic systems to your local MLX models.

## Core Integration Concept

MLX Router exposes an OpenAI-compatible API at `http://localhost:8800/v1` (default configuration). Any framework or client that supports OpenAI's chat completions API can connect to MLX Router by:

1. **Setting the base URL** to your MLX Router instance (`http://localhost:8800/v1`)
2. **Providing a dummy API key** (required by OpenAI clients, but ignored by MLX Router)
3. **Specifying the model ID** from your MLX Router configuration

## Current Limitations

MLX Router v2.0+ has the following limitations that affect agent integrations:

- **No response streaming**: The server returns complete responses, not streaming chunks
- **No tool/function calling**: MLX models don't support structured function calls
- **No structured output**: JSON schema-guided output is not supported

*Future planned releases of mlx-router will provide these features*

The examples below include workarounds for these limitations where applicable.

## Framework Examples

### Microsoft Semantic Kernel Integration

Semantic Kernel works seamlessly with MLX Router using its native OpenAI connector. The integration requires only changing the base URL to point to your local MLX Router instance.

**Setup Instructions:**

1. **Install dependencies:**
   ```bash
   uv pip install semantic-kernel openai
   ```

2. **Navigate to the example directory:**
   ```bash
   cd agents/
   ```

3. **Run the example:**
   ```bash
   python ollama_chat_completion.py
   ```

**Key Integration Points:**
- Uses `OpenAIChatCompletion` connector with custom `base_url`
- Tested with semantic-kernel version 1.35.0
- Based on Microsoft's [ollama_chat_completion.py example](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/local_models/ollama_chat_completion.py)


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

Strands requires a custom model provider to integrate with MLX Router. The implementation includes a workaround for MLX Router's lack of streaming support by simulating streaming events.

**Setup Instructions:**

1. **Install dependencies:**
   ```bash
   cd agents/strands/
   uv pip install -r requirements.txt
   ```

2. **Run the basic agent example:**
   ```bash
   python minimal_agent.py
   ```

3. **Test conversation memory:**
   ```bash
   python conversation_memory.py
   ```

**Key Integration Points:**
- Custom [`MLXModel`](agents/strands/mlx_model.py) provider that implements Strands' `Model` interface
- **Simulated streaming**: Due to MLX Router's non-streaming nature, the provider buffers the complete response and yields it as stream events
- Message format conversion between Strands and OpenAI formats
- Error handling for context window overflow (413 errors)

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

LangChain can be integrated with MLX Router using its OpenAI chat model with a custom base URL.

**Setup Instructions:**

1. **Install dependencies:**
   ```bash
   uv pip install langchain langchain-openai
   ```

2. **Basic usage example:**
   ```python
   from langchain_openai import ChatOpenAI
   from langchain.schema import HumanMessage
   
   # Initialize LangChain with MLX Router
   llm = ChatOpenAI(
       model="mlx-community/Qwen3-30B-A3B-8bit",  # Your model ID
       base_url="http://localhost:8800/v1",
       api_key="dummy-key",  # Required but ignored
       temperature=0.7,
       max_tokens=1000
   )
   
   # Test the connection
   response = llm.invoke([HumanMessage(content="Hello, how are you?")])
   print(response.content)
   ```

**Key Integration Points:**
- Uses `ChatOpenAI` with custom `base_url` parameter
- Supports LangChain's conversation chains and memory systems
- Compatible with LangChain agents (note: function calling limitations apply)

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

### Integration Tips

When integrating any OpenAI-compatible client:

1. **Always set the base URL** to your MLX Router instance
2. **Use any string as API key** (required by clients, ignored by MLX Router)  
3. **Check model IDs** in your `config.json` or via `GET /models` endpoint
4. **Expect non-streaming responses** - some clients may show delayed output [1]
5. **Function calling won't work** - disable tool/function features in the client [1]

*[1] Future planned releases of mlx-router will provide these features*

These integrations enable full-featured, locally running AI systems with rich UIs and agentic capabilities while maintaining complete data privacy.