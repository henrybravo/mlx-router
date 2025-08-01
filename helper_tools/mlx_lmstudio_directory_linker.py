#!/usr/bin/env python3
"""
MLX to LM Studio Directory Linker
Creates clean directory symlinks for MLX models in LM Studio directory
This approach symlinks entire snapshot directories with readable names
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
    """Get list of downloaded MLX models with snapshot paths"""
    hf_cache = Path.home() / ".cache/huggingface/hub"
    
    if not hf_cache.exists():
        return []
    
    models = []
    for model_dir in hf_cache.glob("models--*"):
        if "mlx-community" in model_dir.name or "deepseek" in model_dir.name:
            # Extract readable model name
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            
            # Find latest snapshot directory
            snapshots = list(model_dir.glob("snapshots/*"))
            if snapshots:
                # Sort by modification time, get latest
                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                
                models.append({
                    "name": model_name,
                    "readable_name": model_name.split("/")[-1],  # Just the model name part
                    "publisher": model_name.split("/")[0] if "/" in model_name else "unknown",
                    "snapshot_path": latest_snapshot,
                    "hf_dir": model_dir
                })
    
    return models


def get_main_model_file(snapshot_path):
    """Find the main model file in snapshot"""
    model_files = []
    for file_path in snapshot_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.safetensors':
            if not file_path.name.endswith('.index.json'):
                model_files.append(file_path)
    
    if model_files:
        # Return the largest file (main model)
        return max(model_files, key=lambda f: f.stat().st_size)
    return None


def get_file_size(file_path):
    """Get human readable file size"""
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def create_lmstudio_config(model_info, main_model_file):
    """Create LM Studio compatible model config"""
    config = {
        "name": model_info["readable_name"],
        "publisher": model_info["publisher"],
        "size": get_file_size(main_model_file) if main_model_file else "unknown",
        "description": f"MLX model: {model_info['name']}",
        "filename": main_model_file.name if main_model_file else "model.safetensors",
        "url": f"https://huggingface.co/{model_info['name']}",
        "downloaded": True,
        "isLocal": True,
        "path": str(model_info["snapshot_path"])
    }
    return config


def link_model_directory(model_info, lmstudio_path):
    """Create clean directory symlink for model in LM Studio directory"""
    
    # Create clean directory name: publisher/model-name
    if model_info["publisher"] != "unknown":
        # Create publisher directory if it doesn't exist
        publisher_dir = lmstudio_path / model_info["publisher"]
        publisher_dir.mkdir(exist_ok=True)
        target_dir = publisher_dir / model_info["readable_name"]
    else:
        # Fallback to flat structure
        target_dir = lmstudio_path / model_info["readable_name"]
    
    try:
        # Remove existing link if present
        if target_dir.exists() or target_dir.is_symlink():
            if target_dir.is_symlink():
                target_dir.unlink()
            elif target_dir.is_dir():
                import shutil
                shutil.rmtree(target_dir)
            else:
                target_dir.unlink()
        
        # Create directory symlink to snapshot
        target_dir.symlink_to(model_info["snapshot_path"])
        
        # Find main model file for metadata
        main_model_file = get_main_model_file(model_info["snapshot_path"])
        
        # Create LM Studio config in the symlinked directory
        config_file = target_dir / "lmstudio.json"
        config = create_lmstudio_config(model_info, main_model_file)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Count files in snapshot
        file_count = len([f for f in model_info["snapshot_path"].iterdir() if f.is_file()])
        
        print(f"✅ Linked: {model_info['name']} → {target_dir.relative_to(lmstudio_path)} ({file_count} files)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to link {model_info['name']}: {e}")
        return False


def unlink_model_directory(model_name, lmstudio_path):
    """Remove model directory symlink from LM Studio"""
    import shutil
    
    # Try to find the model in publisher subdirectories
    found_paths = []
    
    # Check flat structure first
    flat_name = model_name.split("/")[-1]
    flat_path = lmstudio_path / flat_name
    if flat_path.exists():
        found_paths.append(flat_path)
    
    # Check publisher/model structure
    if "/" in model_name:
        publisher, model = model_name.split("/", 1)
        structured_path = lmstudio_path / publisher / model
        if structured_path.exists():
            found_paths.append(structured_path)
    
    if not found_paths:
        print(f"⚠️  Model not found: {model_name}")
        return False
    
    try:
        for path in found_paths:
            if path.is_symlink():
                path.unlink()
                print(f"🗑️  Unlinked: {model_name} ({path.relative_to(lmstudio_path)})")
            elif path.is_dir():
                shutil.rmtree(path)
                print(f"🗑️  Removed: {model_name} ({path.relative_to(lmstudio_path)})")
        return True
    except Exception as e:
        print(f"❌ Failed to unlink {model_name}: {e}")
        return False


def list_directory_links(lmstudio_path):
    """List current model directory links"""
    links = []
    
    if not lmstudio_path.exists():
        return links
    
    def scan_directory(directory, prefix=""):
        for item in directory.iterdir():
            if item.name.startswith('.'):
                continue
                
            if item.is_dir() and not item.is_symlink():
                # Scan subdirectories (publisher folders)
                scan_directory(item, f"{item.name}/")
            elif item.is_symlink() and item.is_dir():
                # This is a model symlink
                model_name = f"{prefix}{item.name}"
                target = item.readlink()
                
                # Check if target exists and count files
                file_count = 0
                if target.exists():
                    file_count = len([f for f in target.iterdir() if f.is_file()])
                
                links.append({
                    "name": model_name,
                    "link": item,
                    "target": target,
                    "valid": target.exists(),
                    "files": file_count
                })
    
    scan_directory(lmstudio_path)
    return links


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("MLX to LM Studio Directory Linker")
        print("Creates clean publisher/model directory structure")
        print()
        print("Usage:")
        print("  python3 mlx_lmstudio_directory_linker.py list")
        print("  python3 mlx_lmstudio_directory_linker.py link-all")
        print("  python3 mlx_lmstudio_directory_linker.py link <model_name>")
        print("  python3 mlx_lmstudio_directory_linker.py unlink <model_name>")
        print("  python3 mlx_lmstudio_directory_linker.py status")
        return
    
    command = sys.argv[1]
    lmstudio_path = get_lmstudio_path()
    
    print(f"📁 LM Studio path: {lmstudio_path}")
    
    if command == "list":
        models = get_mlx_models()
        print(f"\n📋 Found {len(models)} MLX models:")
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model['name']} → {model['publisher']}/{model['readable_name']}")
    
    elif command == "status":
        links = list_directory_links(lmstudio_path)
        print(f"\n🔗 Current directory links ({len(links)}):")
        for link in links:
            status = "✅" if link["valid"] else "❌"
            print(f"  {status} {link['name']} ({link['files']} files)")
    
    elif command == "link-all":
        models = get_mlx_models()
        linked = 0
        for model in models:
            if link_model_directory(model, lmstudio_path):
                linked += 1
        print(f"\n✅ Linked {linked}/{len(models)} models")
    
    elif command == "link" and len(sys.argv) == 3:
        model_name = sys.argv[2]
        models = get_mlx_models()
        
        for model in models:
            if model["name"] == model_name:
                link_model_directory(model, lmstudio_path)
                break
        else:
            print(f"❌ Model not found: {model_name}")
    
    elif command == "unlink" and len(sys.argv) == 3:
        model_name = sys.argv[2]
        unlink_model_directory(model_name, lmstudio_path)
    
    else:
        print("❌ Invalid command. Use without arguments for help.")


if __name__ == "__main__":
    main()