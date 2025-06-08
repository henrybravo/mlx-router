#!/usr/bin/env python3
"""
MLX to LM Studio Linker
Creates symlinks for MLX models in LM Studio directory
"""

import json
from pathlib import Path

def get_lmstudio_path():
    """Find LM Studio models directory"""
    home = Path.home()
    
    # Common LM Studio paths
    paths = [
        home / ".cache/lm-studio/models",
        home / "Library/Application Support/LMStudio/models",
        home / ".lmstudio/models"
    ]
    
    for path in paths:
        if path.exists():
            return path
    
    # Create default if none exist
    default_path = home / ".lmstudio/models"
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


def get_mlx_models():
    """Get list of downloaded MLX models"""
    hf_cache = Path.home() / ".cache/huggingface/hub"
    
    if not hf_cache.exists():
        return []
    
    models = []
    for model_dir in hf_cache.glob("models--*"):
        if "mlx-community" in model_dir.name or "deepseek" in model_dir.name:
            # Extract model name
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            
            # Find snapshot directory
            snapshots = list(model_dir.glob("snapshots/*"))
            if snapshots:
                models.append({
                    "name": model_name,
                    "path": snapshots[0]  # Use latest snapshot
                })
    
    return models


def create_lmstudio_config(model_name, model_path):
    """Create LM Studio compatible model config"""
    config = {
        "name": model_name.split("/")[-1],
        "publisher": model_name.split("/")[0] if "/" in model_name else "unknown",
        "size": "unknown",
        "description": f"MLX model: {model_name}",
        "filename": "model.safetensors",
        "url": f"https://huggingface.co/{model_name}",
        "downloaded": True,
        "isLocal": True,
        "path": str(model_path)
    }
    return config


def link_model(model_name, model_path, lmstudio_path):
    """Create symlink for model in LM Studio directory"""
    
    # Clean model name for directory
    safe_name = model_name.replace("/", "--")
    target_dir = lmstudio_path / safe_name
    
    try:
        # Remove existing link if present
        if target_dir.exists() or target_dir.is_symlink():
            if target_dir.is_symlink():
                target_dir.unlink()
            else:
                print(f"⚠️  Directory exists (not a symlink): {target_dir}")
                return False
        
        # Create symlink
        target_dir.symlink_to(model_path)
        
        # Create LM Studio config
        config_file = target_dir / "lmstudio.json"
        config = create_lmstudio_config(model_name, model_path)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Linked: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to link {model_name}: {e}")
        return False


def unlink_model(model_name, lmstudio_path):
    """Remove symlink from LM Studio directory"""
    safe_name = model_name.replace("/", "--")
    target_dir = lmstudio_path / safe_name
    
    try:
        if target_dir.is_symlink():
            target_dir.unlink()
            print(f"🗑️  Unlinked: {model_name}")
            return True
        else:
            print(f"⚠️  Not a symlink: {model_name}")
            return False
    except Exception as e:
        print(f"❌ Failed to unlink {model_name}: {e}")
        return False


def list_links(lmstudio_path):
    """List current MLX model links"""
    links = []
    
    for item in lmstudio_path.iterdir():
        if item.is_symlink() and ("mlx-community" in item.name or "deepseek" in item.name):
            model_name = item.name.replace("--", "/")
            target = item.readlink()
            links.append({
                "name": model_name,
                "link": item,
                "target": target,
                "valid": target.exists()
            })
    
    return links


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("MLX to LM Studio Linker")
        print("Usage:")
        print("  python3 mlx_lmstudio_linker.py list")
        print("  python3 mlx_lmstudio_linker.py link-all")
        print("  python3 mlx_lmstudio_linker.py link <model_name>")
        print("  python3 mlx_lmstudio_linker.py unlink <model_name>")
        print("  python3 mlx_lmstudio_linker.py status")
        return
    
    command = sys.argv[1]
    lmstudio_path = get_lmstudio_path()
    
    print(f"📁 LM Studio path: {lmstudio_path}")
    
    if command == "list":
        models = get_mlx_models()
        print(f"\n📋 Found {len(models)} MLX models:")
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model['name']}")
    
    elif command == "status":
        links = list_links(lmstudio_path)
        print(f"\n🔗 Current links ({len(links)}):")
        for link in links:
            status = "✅" if link["valid"] else "❌"
            print(f"  {status} {link['name']}")
    
    elif command == "link-all":
        models = get_mlx_models()
        linked = 0
        for model in models:
            if link_model(model["name"], model["path"], lmstudio_path):
                linked += 1
        print(f"\n✅ Linked {linked}/{len(models)} models")
    
    elif command == "link" and len(sys.argv) == 3:
        model_name = sys.argv[2]
        models = get_mlx_models()
        
        for model in models:
            if model["name"] == model_name:
                link_model(model["name"], model["path"], lmstudio_path)
                break
        else:
            print(f"❌ Model not found: {model_name}")
    
    elif command == "unlink" and len(sys.argv) == 3:
        model_name = sys.argv[2]
        unlink_model(model_name, lmstudio_path)
    
    else:
        print("❌ Invalid command. Use --help for usage.")


if __name__ == "__main__":
    main()