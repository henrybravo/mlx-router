#!/usr/bin/env python3
"""
MLX Model Downloader
Downloads and verifies MLX models from Hugging Face
"""

import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from mlx_lm import load
import gc

def download_model(model_name, verify=True, force_redownload=False):
    """Download and optionally verify an MLX model"""
    
    print(f"🔍 Checking status for {model_name}...", flush=True)
    status, model_path, incomplete_files = get_model_status(model_name)
    
    if status == "complete" and not force_redownload:
        print(f"✅ Model {model_name} already downloaded and complete", flush=True)
        if verify:
            print("🔍 Verifying model loads correctly...", flush=True)
            try:
                model, tokenizer = load(model_name)
                print(f"✅ Model verified: {model_name}", flush=True)
                
                # Clean up memory
                del model, tokenizer
                gc.collect()
                return True
            except Exception as e:
                if "glm4_moe" in str(e).lower():
                    print(f"⚠️  Model type 'glm4_moe' not supported by mlx_lm. Skipping verification.", flush=True)
                    print(f"ℹ️  Model files are downloaded and may be usable with other tools.", flush=True)
                    return True
                print(f"❌ Model verification failed: {e}", flush=True)
                print(f"ℹ️  Model files are downloaded but may be corrupted. Use 'force_redownload=True' to retry.", flush=True)
                return False
        else:
            return True
    
    if status == "incomplete":
        print(f"⚠️  Found incomplete download with {len(incomplete_files)} partial files", flush=True)
        print(f"🔄 Attempting to resume download for {model_name}", flush=True)
    
    print(f"🔄 Downloading: {model_name}", flush=True)
    
    try:
        # Download model files
        local_path = snapshot_download(
            repo_id=model_name,
            local_files_only=False,
            force_download=force_redownload
        )
        print(f"✅ Downloaded to: {local_path}", flush=True)
        
        # Verify final status
        final_status, _, _ = get_model_status(model_name)
        if final_status != "complete":
            print(f"⚠️  Download may be incomplete (status: {final_status})", flush=True)
        
        if verify:
            print("🔍 Verifying model loads correctly...", flush=True)
            try:
                model, tokenizer = load(model_name)
                print(f"✅ Model verified: {model_name}", flush=True)
                
                # Clean up memory
                del model, tokenizer
                gc.collect()
                
            except Exception as e:
                if "glm4_moe" in str(e).lower():
                    print(f"⚠️  Model type 'glm4_moe' not supported by mlx_lm. Skipping verification.", flush=True)
                    print(f"ℹ️  Model files are downloaded and may be usable with other tools.", flush=True)
                    return True
                print(f"❌ Model verification failed: {e}", flush=True)
                print(f"ℹ️  Model files are downloaded but may be corrupted. Use 'force_redownload=True' to retry.", flush=True)
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}", flush=True)
        return False


def get_cache_path():
    """Get Hugging Face cache directory"""
    try:
        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        return cache_path
    except Exception as e:
        print(f"❌ Error accessing cache path: {e}", flush=True)
        return None

def get_model_status(model_name):
    """Check if model is downloaded and its status"""
    cache_path = get_cache_path()
    if not cache_path:
        return "error", None, []
    
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = cache_path / model_dir_name
    
    if not model_path.exists():
        return "not_downloaded", None, []
    
    blobs_path = model_path / "blobs"
    if not blobs_path.exists():
        return "no_blobs", model_path, []
    
    # Check for incomplete files
    incomplete_files = list(blobs_path.glob("*.incomplete"))
    all_files = list(blobs_path.glob("*"))
    
    if incomplete_files:
        return "incomplete", model_path, incomplete_files
    
    # Check if snapshots exist
    snapshots_path = model_path / "snapshots"
    if snapshots_path.exists() and list(snapshots_path.glob("*")):
        return "complete", model_path, all_files
    
    return "unknown", model_path, all_files

def clean_incomplete_model(model_name):
    """Clean incomplete downloads for a model"""
    status, _, files = get_model_status(model_name)
    incomplete_files = files if status == "incomplete" else []
    
    if status == "incomplete":
        print(f"🧹 Cleaning {len(incomplete_files)} incomplete files for {model_name}", flush=True)
        for file_path in incomplete_files:
            try:
                file_path.unlink()
                print(f"   Removed: {file_path.name}", flush=True)
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path.name}: {e}", flush=True)
        return True
    elif status == "not_downloaded":
        print(f"ℹ️  Model {model_name} not downloaded yet", flush=True)
        return False
    else:
        print(f"✅ Model {model_name} appears complete", flush=True)
        return False

def remove_model(model_name):
    """Completely remove a model from cache"""
    status, model_path, _ = get_model_status(model_name)
    
    if status == "not_downloaded":
        print(f"ℹ️  Model {model_name} not found in cache", flush=True)
        return False
    
    try:
        shutil.rmtree(model_path)
        print(f"🗑️  Completely removed {model_name} from cache", flush=True)
        return True
    except Exception as e:
        print(f"❌ Failed to remove {model_name}: {e}", flush=True)
        return False

def discover_local_models():
    """Discover all downloaded models in cache directory"""
    cache_path = get_cache_path()
    if not cache_path or not cache_path.exists():
        return []
    
    models = []
    # Look for model directories (format: models--org--model)
    for model_dir in cache_path.glob("models--*"):
        if model_dir.is_dir():
            # Convert directory name back to model name format
            dir_name = model_dir.name
            if dir_name.startswith("models--"):
                model_name = dir_name[8:]  # Remove "models--" prefix
                parts = model_name.split("--")
                if len(parts) >= 2:
                    org = parts[0]
                    model_parts = parts[1:]
                    model_name = f"{org}/{'-'.join(model_parts)}"
                    models.append(model_name)
    
    return sorted(models)

def list_mlx_models():
    """Show all downloaded MLX models with status"""
    
    models = discover_local_models()
    
    if not models:
        print("📋 No MLX models found in cache directory", flush=True)
        print(f"   Cache path: {get_cache_path() or 'unavailable'}", flush=True)
        print("   Try downloading a model with: python3 mlx_downloader.py <model_name>", flush=True)
        return models
    
    print(f"📋 Downloaded MLX Models ({len(models)} found):", flush=True)
    for i, model in enumerate(models, 1):
        status, _, files = get_model_status(model)
        incomplete_files = files if status == "incomplete" else []
        status_emoji = {
            "complete": "✅",
            "incomplete": "⚠️ ",
            "not_downloaded": "⬜",
            "no_blobs": "❓",
            "unknown": "❓",
            "error": "❌"
        }
        
        status_text = {
            "complete": "Complete",
            "incomplete": f"Incomplete ({len(incomplete_files)} partial files)",
            "not_downloaded": "Not downloaded",
            "no_blobs": "No blobs",
            "unknown": "Unknown status",
            "error": "Error accessing cache"
        }
        
        print(f"  {i:2d}. {status_emoji[status]} {model} - {status_text[status]}", flush=True)
    
    return models


def main():
    try:
        if len(sys.argv) < 2:
            print("Enhanced MLX Model Downloader", flush=True)
            print("Usage:", flush=True)
            print("  python3 mlx_downloader.py <model_name>        # Download or resume specific model", flush=True)
            print("  python3 mlx_downloader.py list                # List downloaded models with status", flush=True)
            print("  python3 mlx_downloader.py download <num>      # Download or resume by number from list", flush=True)
            print("  python3 mlx_downloader.py status <model|num>  # Check model status", flush=True)
            print("  python3 mlx_downloader.py clean <model|num>   # Clean incomplete files", flush=True)
            print("  python3 mlx_downloader.py remove <model|num>  # Remove model completely", flush=True)
            print("  python3 mlx_downloader.py clean-all           # Clean all incomplete files", flush=True)
            print("", flush=True)
            list_mlx_models()
            return
        
        command = sys.argv[1]
        
        if command == "list":
            list_mlx_models()
            
        elif command == "download" and len(sys.argv) == 3:
            try:
                models = discover_local_models()
                if not models:
                    print("❌ No models found in cache directory. Use direct model name to download new models.", flush=True)
                    return
                
                index = int(sys.argv[2]) - 1
                if 0 <= index < len(models):
                    model_name = models[index]
                    download_model(model_name)
                else:
                    print(f"❌ Invalid number. Choose 1-{len(models)}", flush=True)
            except ValueError:
                print("❌ Please provide a valid number", flush=True)
        
        elif command == "status" and len(sys.argv) == 3:
            model_arg = sys.argv[2]
            
            try:
                index = int(model_arg) - 1
                models = discover_local_models()
                if not models:
                    print("❌ No models found in cache directory", flush=True)
                    return
                
                if 0 <= index < len(models):
                    model_name = models[index]
                else:
                    print(f"❌ Invalid number. Choose 1-{len(models)}", flush=True)
                    return
            except ValueError:
                model_name = model_arg
            
            status, model_path, files = get_model_status(model_name)
            print(f"📊 Status for {model_name}: {status}", flush=True)
            if model_path:
                print(f"   Path: {model_path}", flush=True)
            if files:
                print(f"   Files: {len(files)} files", flush=True)
                if status == "incomplete":
                    print(f"   Incomplete files: {[f.name for f in files]}", flush=True)
        
        elif command == "clean" and len(sys.argv) == 3:
            model_arg = sys.argv[2]
            
            try:
                index = int(model_arg) - 1
                models = discover_local_models()
                if not models:
                    print("❌ No models found in cache directory", flush=True)
                    return
                
                if 0 <= index < len(models):
                    model_name = models[index]
                    clean_incomplete_model(model_name)
                else:
                    print(f"❌ Invalid number. Choose 1-{len(models)}", flush=True)
            except ValueError:
                model_name = model_arg
                clean_incomplete_model(model_name)
        
        elif command == "remove" and len(sys.argv) == 3:
            model_arg = sys.argv[2]
            
            try:
                index = int(model_arg) - 1
                models = discover_local_models()
                if not models:
                    print("❌ No models found in cache directory", flush=True)
                    return
                
                if 0 <= index < len(models):
                    model_name = models[index]
                else:
                    print(f"❌ Invalid number. Choose 1-{len(models)}", flush=True)
                    return
            except ValueError:
                model_name = model_arg
            
            confirm = input(f"⚠️  Are you sure you want to completely remove {model_name}? (y/N): ")
            if confirm.lower() == 'y':
                remove_model(model_name)
            else:
                print("❌ Cancelled", flush=True)
        
        elif command == "clean-all":
            models = discover_local_models()
            if not models:
                print("❌ No models found in cache directory", flush=True)
                return
            
            cleaned_count = 0
            for model in models:
                if clean_incomplete_model(model):
                    cleaned_count += 1
            print(f"🧹 Cleaned incomplete files for {cleaned_count} models", flush=True)
                
        else:
            model_name = sys.argv[1]
            download_model(model_name)
    
    except Exception as e:
        print(f"❌ Script failed: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()