# Microsoft Agent Framework samples for mlx-router

These five samples mirror the upstream declarative examples but are wired to mlx-router via the OpenAI-compatible interface.

## Setup
1) Install deps (Python 3.11+): `uv pip install -r requirements.txt`
2) Environment (defaults shown):
   - `MLX_ROUTER_BASE_URL`=`http://localhost:8800/v1`
   - `MLX_ROUTER_API_KEY`=`dummy-key` (or your key if enabled)
   - `MLX_ROUTER_MODEL`=`mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` (or any loaded model)
   - The scripts also populate `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` from these values for Agent Framework compatibility.

Environment loading order (what takes precedence):
1. Explicit environment variables in the shell (highest priority)
2. Values from `.env` in this folder (loaded via `python-dotenv`)
3. Hardcoded fallbacks in the scripts (lowest priority)

If you want the `.env` values to win, avoid exporting conflicting variables in the shell. If you want the shell to win, export them before running the scripts.

## Samples
All scripts live under `agents/`:
- `get_weather_agent.py`
- `microsoft_learn_agent.py`
- `inline_yaml.py`
- `azure_openai_responses_agent.py`
- `openai_responses_agent.py`

YAML declarations are under `agent-declarations/` and have been normalized to use OpenAI-style `key` connections and `=Env.OPENAI_MODEL`.

### Unused declarations (extras)
The following YAMLs are provided but not wired into the current `agents/*.py` runners. You can use them by creating a new runner (copy one of the existing scripts) and pointing to the YAML path under `agent-declarations/`:
- `OpenAI.yaml`
- `OpenAIAssistants.yaml`
- `OpenAIChat.yaml`

To use one, modify a runner to load the desired YAML, apply the same `=Env.OPENAI_*` substitutions (model/key/base URL), and run the script.

## Running
Ensure mlx-router is running, then from this folder:
```bash
python agents/get_weather_agent.py
```
Swap the filename for other samples. Responses stream through mlx-router using SSE.
