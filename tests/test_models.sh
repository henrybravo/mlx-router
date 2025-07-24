#!/bin/bash

mlx_host="http://localhost:8888"
echo "curl -s ${mlx_host}/health | jq"
curl -s ${mlx_host}/health | jq

echo "curl -s ${mlx_host}/v1/models | jq"
curl -s ${mlx_host}/v1/models | jq

# mlx-community/Llama-3.2-3B-Instruct-4bit
echo "curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H \"Content-Type: application/json\" \
     -d '{\"model\": \"mlx-community/Llama-3.2-3B-Instruct-4bit\", \"messages\": [{\"role\": \"user\", \"content\": \"provide info about yourself as a llm model\"}]}'"
curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit", "messages": [{"role": "user", "content": "provide info about yourself as a llm model"}]}' |jq

# deepseek-coder-6.7b-instruct
echo "curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H \"Content-Type: application/json\" \
     -d '{\"model\": \"deepseek-ai/deepseek-coder-6.7b-instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"provide info about yourself as a llm model\"}]}'"
curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-ai/deepseek-coder-6.7b-instruct", "messages": [{"role": "user", "content": "Provide info about yourself as a llm model"}]}' | jq

# phi-4-reasoning-plus-6bit
echo "curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H \"Content-Type: application/json\" \
     -d '{\"model\": \"mlx-community/Phi-4-reasoning-plus-6bit\", \"messages\": [{\"role\": \"user\", \"content\": \"provide info about yourself as a llm model\"}]}'"
curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "mlx-community/Phi-4-reasoning-plus-6bit", "messages": [{"role": "user", "content": "Provide info about yourself as a llm model"}]}' | jq

# qwen3-30b-a3b-8bit
echo "curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H \"Content-Type: application/json\" \
     -d '{\"model\": \"mlx-community/Qwen3-30B-A3B-8bit\", \"messages\": [{\"role\": \"user\", \"content\": \"provide info about yourself as a llm model\"}]}'"
curl -s -X POST ${mlx_host}/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "mlx-community/Qwen3-30B-A3B-8bit", "messages": [{"role": "user", "content": "Provide info about yourself as a llm model"}]}' | jq
