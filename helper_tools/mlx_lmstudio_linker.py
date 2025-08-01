#!/usr/bin/env python3
"""
MLX to LM Studio Linker
Creates symlinks for MLX models in LM Studio directory
"""

import json
import os
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


def find_model_files(snapshot_path):
    """Find actual model files in snapshot directory"""
    model_files = []
    
    # Look for safetensors files (MLX format)
    for file_path in snapshot_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.safetensors':
            # Skip index files, get actual model files
            if not file_path.name.endswith('.index.json'):
                model_files.append(file_path)
    
    return model_files


def get_file_size(file_path):
    """Get human readable file size"""
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def create_lmstudio_config(model_name, model_path, main_model_file):
    """Create LM Studio compatible model config"""
    config = {
        "name": model_name.split("/")[-1],
        "publisher": model_name.split("/")[0] if "/" in model_name else "unknown",
        "size": get_file_size(main_model_file),
        "description": f"MLX model: {model_name}",
        "filename": main_model_file.name,
        "url": f"https://huggingface.co/{model_name}",
        "downloaded": True,
        "isLocal": True,
        "path": str(model_path)
    }
    return config


def link_model(model_name, model_path, lmstudio_path):
    """Create symlinks for model files in LM Studio directory"""
    
    # Clean model name for directory
    safe_name = model_name.replace("/", "--")
    target_dir = lmstudio_path / safe_name
    
    try:
        # Remove existing directory if present
        if target_dir.exists():
            if target_dir.is_symlink():
                target_dir.unlink()
            elif target_dir.is_dir():
                # Remove directory and all its contents
                import shutil
                shutil.rmtree(target_dir)
            else:
                target_dir.unlink()
        
        # Create target directory
        target_dir.mkdir(parents=True)
        
        # Find model files in snapshot
        model_files = find_model_files(model_path)
        if not model_files:
            print(f"⚠️  No safetensors files found in {model_path}")
            return False
        
        # Find the main model file (largest or first if same size)
        main_model_file = max(model_files, key=lambda f: f.stat().st_size)
        
        # Create symlinks for all files in the snapshot
        linked_files = []
        for file_path in model_path.iterdir():
            if file_path.is_file():
                link_path = target_dir / file_path.name
                # Create absolute symlink to avoid broken relative paths
                link_path.symlink_to(file_path.resolve())
                linked_files.append(file_path.name)
        
        # Create LM Studio config
        config_file = target_dir / "lmstudio.json"
        config = create_lmstudio_config(model_name, model_path, main_model_file)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Linked: {model_name} ({len(linked_files)} files, main: {main_model_file.name})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to link {model_name}: {e}")
        return False


def unlink_model(model_name, lmstudio_path):
    """Remove model directory from LM Studio"""
    import shutil
    
    safe_name = model_name.replace("/", "--")
    target_dir = lmstudio_path / safe_name
    
    try:
        if target_dir.exists():
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
                print(f"🗑️  Unlinked: {model_name}")
                return True
            elif target_dir.is_symlink():
                target_dir.unlink()
                print(f"🗑️  Unlinked: {model_name}")
                return True
            else:
                target_dir.unlink()
                print(f"🗑️  Unlinked: {model_name}")
                return True
        else:
            print(f"⚠️  Model not linked: {model_name}")
            return False
    except Exception as e:
        print(f"❌ Failed to unlink {model_name}: {e}")
        return False


def list_links(lmstudio_path):
    """List current MLX model links"""
    links = []
    
    if not lmstudio_path.exists():
        return links
    
    for item in lmstudio_path.iterdir():
        if item.is_dir() and ("mlx-community" in item.name or "deepseek" in item.name):
            model_name = item.name.replace("--", "/")
            
            # Check if directory contains symlinked files
            symlink_count = 0
            broken_count = 0
            
            for file_path in item.iterdir():
                if file_path.is_symlink():
                    symlink_count += 1
                    if not file_path.exists():
                        broken_count += 1
            
            valid = symlink_count > 0 and broken_count == 0
            
            links.append({
                "name": model_name,
                "link": item,
                "target": item,
                "valid": valid,
                "symlinks": symlink_count,
                "broken": broken_count
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
            detail = f"({link['symlinks']} files"
            if link['broken'] > 0:
                detail += f", {link['broken']} broken"
            detail += ")"
            print(f"  {status} {link['name']} {detail}")
    
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