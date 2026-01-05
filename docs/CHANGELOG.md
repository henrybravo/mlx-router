# Changelog

## [v2.3.0] - 2026-01-04

### Added
- **Vision Model Support**
  - Image processing - Send PNG, JPEG, WebP, BMP images for analysis via OpenAI-compatible multimodal format
  - PDF Support - Automatic PDF-to-image conversion for document OCR (requires `poppler`)
  - mlx-vlm Integration - Powered by [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for efficient vision inference
  - Config-based detection - Use `"supports_vision": true` to enable vision capabilities per model
  - Tested with [chandra-8bit](https://huggingface.co/mlx-community/chandra-8bit) OCR model
  - Client compatibility - Works with curl, OpenWebUI, and Python clients

## [v2.1.3] - 2025-09-26

### Added
- Support for `swap_critical_percent` and `swap_high_percent` configuration options
- **Enhanced Response Streaming**
  - Multiple streaming formats - Choose between SSE, JSON Lines, or JSON Array formats
  - Goose/OpenWebUI compatibility - JSON Array format works with clients that use `response.json()`
  - Real-time token delivery - 90%+ reduction in time-to-first-token
  - Server-Sent Events - OpenAI-compatible streaming format (default)
  - Async generators - Non-blocking streaming with FastAPI
  - Memory efficient - Reduced peak memory usage during generation

## [v2.1.2]

### Added
- **Custom Model Directory Loading**
  - Local model support - Load models from any local directory (default: `$HOME/models`)
  - Automatic discovery - Models placed in configured directory are automatically detected
  - Dynamic configuration - Model parameters extracted from local `config.json` files
  - HuggingFace cache compatible - Supports both direct directories and HF cache naming
  - Environment variable support - `MLX_MODEL_DIR` for custom directory configuration
  - Hot-swapping - Switch between local models without server restart

## [v2.0.0] - 2025-06-15

### Added
- **🔧 Function Calling**
  - **OpenAI-compatible** - Full compliance with function calling API
  - **Prompt engineering** - Tool instructions injected into model prompts
  - **JSON parsing** - Robust extraction and validation of tool calls
  - **Schema validation** - Tool arguments validated against provided schemas
  - **Error resilience** - Graceful fallback to text responses

- **🚀 Major Improvements**
  - **FastAPI Integration** - Modern async API framework with automatic documentation
  - **Modular Architecture** - Clean separation into config/, core/, and api/ modules
  - **Enhanced Error Handling** - Comprehensive HTTP status codes and error responses
  - **Interactive Documentation** - Built-in Swagger UI and ReDoc interfaces
  - **Improved Performance** - Async request handling and optimized memory management
  - **Better Monitoring** - Enhanced health endpoints with detailed system metrics