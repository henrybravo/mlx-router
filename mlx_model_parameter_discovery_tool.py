#!/usr/bin/env python3
"""
Model Parameter Discovery Tool
Helps find optimal parameters for new MLX models
"""

import sys
import json
from pathlib import Path
import os
import logging
from datetime import datetime
import gc

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
from transformers import AutoTokenizer, AutoConfig

def setup_logging(model_short_name, run_benchmark):
    """Setup comprehensive logging with file and console output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("discovery_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with benchmark indicator
    benchmark_suffix = "_benchmark" if run_benchmark else ""
    log_file = log_dir / f"discovery_{model_short_name}{benchmark_suffix}_{timestamp}.log"
    
    # Setup logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Model Parameter Discovery Tool ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {model_short_name}")
    logger.info(f"Benchmark mode: {run_benchmark}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("=" * 50)
    
    return logger, log_file

def setup_logging(model_short_name, run_benchmark):
    """Setup comprehensive logging with file and console output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("discovery_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with benchmark indicator
    benchmark_suffix = "_benchmark" if run_benchmark else ""
    log_file = log_dir / f"discovery_{model_short_name}{benchmark_suffix}_{timestamp}.log"
    
    # Setup logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Model Parameter Discovery Tool ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model: {model_short_name}")
    logger.info(f"Benchmark mode: {run_benchmark}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("=" * 50)
    
    return logger, log_file

def discover_model_params(model_name, logger):
    """Discover optimal parameters for a model"""
    logger.info(f"\n🔍 Investigating model: {model_name}\n")
    
    results = {
        "model_name": model_name,
        "discovered_params": {},
        "chat_template": None,
        "notes": []
    }
    
    # 1. Try to load config from HuggingFace
    try:
        logger.info("Loading model config from HuggingFace...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✅ Loaded model config from HuggingFace")
        
        # Log all config attributes
        logger.debug("Model config attributes:")
        for attr in dir(config):
            if not attr.startswith('_'):
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        logger.debug(f"  {attr}: {value}")
                except:
                    pass
        
        # Extract useful parameters
        if hasattr(config, 'max_position_embeddings'):
            max_tokens = min(config.max_position_embeddings, 4096)
            results["discovered_params"]["max_tokens"] = max_tokens
            logger.info(f"Max tokens set to: {max_tokens}")
        
        if hasattr(config, 'vocab_size'):
            results["notes"].append(f"Vocab size: {config.vocab_size}")
            logger.info(f"Vocab size: {config.vocab_size}")
            
        if hasattr(config, 'num_hidden_layers'):
            results["notes"].append(f"Layers: {config.num_hidden_layers}")
            logger.info(f"Number of layers: {config.num_hidden_layers}")
            
    except Exception as e:
        logger.warning(f"⚠️  Could not load config: {e}")
    
    # 2. Check tokenizer for chat template
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(tokenizer, 'chat_template'):
            results["chat_template"] = detect_chat_template_type(tokenizer.chat_template)
            logger.info(f"✅ Detected chat template: {results['chat_template']}")
            
            # Log first 500 chars of template for debugging
            template_str = str(tokenizer.chat_template)[:500]
            logger.debug(f"Chat template preview: {template_str}...")
    except Exception as e:
        logger.warning(f"⚠️  Could not load tokenizer: {e}")
    
    # 3. Search for model family patterns
    model_family = detect_model_family(model_name)
    if model_family:
        family_defaults = get_family_defaults(model_family)
        results["discovered_params"].update(family_defaults)
        results["notes"].append(f"Model family: {model_family}")
        logger.info(f"Detected model family: {model_family}")
        logger.debug(f"Family defaults: {json.dumps(family_defaults, indent=2)}")
    
    # 4. Memory requirements estimation
    estimated_memory = estimate_memory_requirements(model_name)
    results["discovered_params"]["required_memory_gb"] = estimated_memory
    logger.info(f"Estimated memory requirement: {estimated_memory}GB")
    
    return results

def detect_chat_template_type(template):
    """Detect the type of chat template from the template string"""
    if not template:
        return "generic"
    
    template_str = str(template).lower()
    
    # Check for known patterns
    if "<|im_start|>" in template_str and "<|im_end|>" in template_str:
        return "chatml"  # ChatML format (Qwen, some others)
    elif "<|begin_of_text|>" in template_str:
        return "llama3"
    elif "### instruction:" in template_str:
        return "deepseek"
    elif "<|user|>" in template_str and "<|assistant|>" in template_str:
        return "phi4"
    elif "[inst]" in template_str:
        return "llama2"
    else:
        return "generic"

def detect_model_family(model_name):
    """Detect model family from name"""
    name_lower = model_name.lower()
    
    families = {
        "llama": ["llama"],
        "qwen": ["qwen"],
        "deepseek": ["deepseek"],
        "phi": ["phi"],
        "mistral": ["mistral"],
        "vicuna": ["vicuna"],
        "wizard": ["wizard"],
        "yi": ["yi-"],
        "mixtral": ["mixtral"],
        "gemma": ["gemma"],
        "starcoder": ["starcoder"],
        "codellama": ["codellama"],
        "virtuoso": ["virtuoso"]
    }
    
    for family, patterns in families.items():
        if any(pattern in name_lower for pattern in patterns):
            return family
    
    return None

def get_family_defaults(family):
    """Get default parameters for known model families"""
    defaults = {
        "llama": {
            "temp": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "max_tokens": 4096
        },
        "qwen": {
            "temp": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "max_tokens": 8192
        },
        "deepseek": {
            "temp": 0.6,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "max_tokens": 4096
        },
        "phi": {
            "temp": 0.3,
            "top_p": 0.9,
            "top_k": 25,
            "min_p": 0.08,
            "max_tokens": 4096
        },
        "virtuoso": {
            # Based on typical creative/artistic models
            "temp": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "min_p": 0.03,
            "max_tokens": 4096
        }
    }
    
    return defaults.get(family, {
        "temp": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "max_tokens": 4096
    })

def estimate_memory_requirements(model_name):
    """Estimate memory requirements based on model name"""
    name_lower = model_name.lower()
    
    # Look for size indicators
    if "70b" in name_lower:
        return 45
    elif "30b" in name_lower or "34b" in name_lower:
        return 20
    elif "13b" in name_lower or "14b" in name_lower:
        return 10
    elif "7b" in name_lower or "8b" in name_lower:
        return 8
    elif "3b" in name_lower or "4b" in name_lower:
        return 4
    elif "1b" in name_lower or "2b" in name_lower:
        return 2
    else:
        # Default for unknown
        return 8

def generate_config_entry(results):
    """Generate a config.json entry for the model"""
    params = results["discovered_params"]
    
    config_entry = {
        results["model_name"]: {
            "max_tokens": params.get("max_tokens", 4096),
            "temp": params.get("temp", 0.7),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 40),
            "min_p": params.get("min_p", 0.05),
            "chat_template": results.get("chat_template", "generic"),
            "required_memory_gb": params.get("required_memory_gb", 8),
            "memory_pressure_max_tokens": {
                "normal": params.get("max_tokens", 4096),
                "moderate": min(params.get("max_tokens", 4096) // 2, 2048),
                "high": min(params.get("max_tokens", 4096) // 4, 1024),
                "critical": min(params.get("max_tokens", 4096) // 8, 512)
            }
        }
    }
    
    return config_entry

def benchmark_model_params(model_name, logger, test_prompts=None):
    """Run quick benchmarks to find optimal parameters"""
    if test_prompts is None:
        test_prompts = [
            "Write a short poem about AI",
            "Explain quantum computing in simple terms",
            "Generate a Python function to sort a list"
        ]
    
    logger.info(f"\n🧪 Running parameter benchmarks for {model_name}...")
    logger.info(f"Test prompts: {test_prompts}")
    
    benchmark_start_time = datetime.now()
    
    try:
        # Import MLX dependencies here to avoid issues if not needed
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        import time
        
        # Load model once
        logger.info("Loading model for benchmarking...")
        load_start = time.time()
        model, tokenizer = load(model_name)
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Log memory usage after loading
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage after model load: {memory_info.rss / 1024**3:.2f}GB")
        except:
            pass
        
        # Results storage
        benchmark_results = {
            "temperature_tests": [],
            "top_p_tests": [],
            "combined_tests": [],
            "optimal_params": {},
            "notes": [],
            "metadata": {
                "start_time": benchmark_start_time.isoformat(),
                "model_load_time": load_time
            }
        }
        
        # Test 1: Temperature variations
        logger.info("\n=== Testing temperature variations ===")
        temp_scores = {}
        for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
            logger.info(f"\nTesting temperature: {temp}")
            scores = []
            
            for i, prompt in enumerate(test_prompts[:2]):  # Use first 2 prompts for speed
                logger.debug(f"  Prompt {i+1}: {prompt[:50]}...")
                start_time = time.time()
                
                try:
                    sampler = make_sampler(temp=temp, top_p=0.9, min_tokens_to_keep=1)
                    response = generate(
                        model, tokenizer,
                        prompt=prompt,
                        max_tokens=100,
                        sampler=sampler,
                        verbose=False
                    )
                    gen_time = time.time() - start_time
                    
                    # Calculate metrics
                    words = response.split()
                    word_count = len(words)
                    unique_words = len(set(words))
                    diversity_score = unique_words / word_count if word_count > 0 else 0
                    
                    # Simple coherence check (ratio of complete sentences)
                    sentences = response.count('.') + response.count('!') + response.count('?')
                    coherence_score = min(1.0, sentences / (word_count / 10)) if word_count > 0 else 0
                    
                    scores.append({
                        "diversity": diversity_score,
                        "coherence": coherence_score,
                        "speed": 1.0 / gen_time if gen_time > 0 else 0,
                        "length": word_count
                    })
                    
                    logger.debug(f"    Response preview: {response[:100]}...")
                    logger.debug(f"    Metrics: diversity={diversity_score:.3f}, coherence={coherence_score:.3f}, time={gen_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"    Error with temp {temp}: {e}")
                    scores.append({"diversity": 0, "coherence": 0, "speed": 0, "length": 0})
            
            # Average scores for this temperature
            if scores:
                avg_diversity = sum(s["diversity"] for s in scores) / len(scores)
                avg_coherence = sum(s["coherence"] for s in scores) / len(scores)
                avg_speed = sum(s["speed"] for s in scores) / len(scores)
                
                # Combined score (weighted)
                combined_score = (avg_diversity * 0.3 + avg_coherence * 0.5 + avg_speed * 0.2)
                temp_scores[temp] = combined_score
                
                temp_result = {
                    "temp": temp,
                    "diversity": avg_diversity,
                    "coherence": avg_coherence,
                    "combined_score": combined_score
                }
                benchmark_results["temperature_tests"].append(temp_result)
                
                logger.info(f"  Temperature {temp} results: diversity={avg_diversity:.3f}, coherence={avg_coherence:.3f}, score={combined_score:.3f}")
        
        # Find optimal temperature
        optimal_temp = max(temp_scores, key=temp_scores.get) if temp_scores else 0.7
        logger.info(f"\n✓ Optimal temperature: {optimal_temp}")
        
        # Test 2: Top-p variations (with optimal temp)
        logger.info(f"\n=== Testing top_p variations (with temp={optimal_temp}) ===")
        top_p_scores = {}
        for top_p in [0.8, 0.85, 0.9, 0.95, 0.99]:
            logger.info(f"Testing top_p: {top_p}")
            
            try:
                sampler = make_sampler(temp=optimal_temp, top_p=top_p, min_tokens_to_keep=1)
                response = generate(
                    model, tokenizer,
                    prompt=test_prompts[0],  # Use one prompt for speed
                    max_tokens=100,
                    sampler=sampler,
                    verbose=False
                )
                
                # Quick quality check
                words = response.split()
                unique_ratio = len(set(words)) / len(words) if words else 0
                top_p_scores[top_p] = unique_ratio
                
                benchmark_results["top_p_tests"].append({
                    "top_p": top_p,
                    "unique_ratio": unique_ratio
                })
                
                logger.debug(f"  Response preview: {response[:100]}...")
                logger.info(f"  Unique word ratio: {unique_ratio:.3f}")
                
            except Exception as e:
                logger.error(f"  Error with top_p {top_p}: {e}")
        
        # Find optimal top_p
        optimal_top_p = max(top_p_scores, key=top_p_scores.get) if top_p_scores else 0.9
        logger.info(f"\n✓ Optimal top_p: {optimal_top_p}")
        
        # Test 3: Combined parameter test with different prompt types
        logger.info(f"\n=== Testing combined parameters on different task types ===")
        task_results = {}
        
        test_configs = {
            "creative": {"temp": min(0.9, optimal_temp + 0.1), "top_p": min(0.99, optimal_top_p + 0.05)},
            "technical": {"temp": max(0.1, optimal_temp - 0.3), "top_p": max(0.8, optimal_top_p - 0.05)},
            "balanced": {"temp": optimal_temp, "top_p": optimal_top_p}
        }
        
        for task_type, config in test_configs.items():
            logger.info(f"\nTesting {task_type} configuration: {config}")
            try:
                # Adjust prompt based on task type
                if task_type == "creative":
                    prompt = "Write a creative story about a magical forest"
                elif task_type == "technical":
                    prompt = "Explain the quicksort algorithm step by step"
                else:
                    prompt = "Describe the benefits of renewable energy"
                
                logger.debug(f"  Prompt: {prompt}")
                
                sampler = make_sampler(
                    temp=config["temp"], 
                    top_p=config["top_p"], 
                    min_tokens_to_keep=1
                )
                
                start_time = time.time()
                response = generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=150,
                    sampler=sampler,
                    verbose=False
                )
                gen_time = time.time() - start_time
                
                task_results[task_type] = {
                    "config": config,
                    "generation_time": gen_time,
                    "response_length": len(response.split()),
                    "response_preview": response[:100] + "..."
                }
                
                logger.info(f"  Generation time: {gen_time:.2f}s")
                logger.info(f"  Response length: {len(response.split())} words")
                logger.debug(f"  Response: {response[:200]}...")
                
            except Exception as e:
                logger.error(f"  Error with {task_type}: {e}")
        
        benchmark_results["combined_tests"] = task_results
        
        # Determine final optimal parameters
        if "creative" in model_name.lower() or "chat" in model_name.lower():
            # Creative/chat models benefit from higher temperature
            final_temp = min(0.9, optimal_temp + 0.1)
            final_top_p = min(0.95, optimal_top_p)
        elif "code" in model_name.lower() or "math" in model_name.lower():
            # Technical models need lower temperature
            final_temp = max(0.1, optimal_temp - 0.2)
            final_top_p = max(0.85, optimal_top_p - 0.05)
        else:
            # Balanced approach
            final_temp = optimal_temp
            final_top_p = optimal_top_p
        
        benchmark_results["optimal_params"] = {
            "temp": round(final_temp, 2),
            "top_p": round(final_top_p, 2),
            "top_k": 40,  # Standard default
            "min_p": 0.05  # Standard default
        }
        
        # Calculate total benchmark time
        total_time = (datetime.now() - benchmark_start_time).total_seconds()
        benchmark_results["metadata"]["total_time"] = total_time
        benchmark_results["metadata"]["end_time"] = datetime.now().isoformat()
        
        benchmark_results["notes"].append(f"Benchmarked on {len(test_prompts)} diverse prompts")
        benchmark_results["notes"].append(f"Best balance achieved with temp={final_temp:.2f}, top_p={final_top_p:.2f}")
        benchmark_results["notes"].append(f"Total benchmark time: {total_time:.2f} seconds")
        
        logger.info(f"\n=== Benchmarking Summary ===")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Optimal temperature: {final_temp:.2f}")
        logger.info(f"Optimal top_p: {final_top_p:.2f}")
        
        # Save detailed benchmark results
        benchmark_file = Path(f"benchmark_results_{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Detailed benchmark results saved to: {benchmark_file}")
        
        # Clean up
        del model
        del tokenizer
        mx.clear_cache()
        gc.collect()
        
        logger.info("✅ Benchmarking complete!")
        
        return benchmark_results
        
    except ImportError as e:
        logger.warning(f"Cannot run benchmarks - MLX not available: {e}")
        return {
            "optimal_params": {
                "temp": 0.7,
                "top_p": 0.9
            },
            "notes": ["Benchmarking skipped - MLX not available, using defaults"]
        }
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return {
            "optimal_params": {
                "temp": 0.7,
                "top_p": 0.9
            },
            "notes": [f"Benchmarking failed: {str(e)}, using defaults"]
        }

def main():
    if len(sys.argv) < 2:
        print("Model Parameter Discovery Tool")
        print("Usage:")
        print("  python3 mlx_model_parameter_discovery_tool.py <model_name>        # Discover parameters")
        print("  python3 mlx_model_parameter_discovery_tool.py <model_name> --benchmark  # With benchmarking\n")
        return

    model_name = sys.argv[1]
    model_short_name = model_name.split('/')[-1]
    run_benchmark = "--benchmark" in sys.argv
    
    # Setup logging
    logger, log_file = setup_logging(model_short_name, run_benchmark)

    try:
        # Discover parameters
        results = discover_model_params(model_name, logger)
        
        # Optionally run benchmarks
        if run_benchmark:
            benchmark_results = benchmark_model_params(model_name, logger)
            
            # Update discovered params with benchmark results
            if "optimal_params" in benchmark_results:
                results["discovered_params"].update(benchmark_results["optimal_params"])
                results["notes"].extend(benchmark_results.get("notes", []))
        
        # Generate config entry
        config_entry = generate_config_entry(results)
        
        logger.info("\n📝 Suggested config.json entry:")
        logger.info(json.dumps(config_entry, indent=4))
        
        logger.info("\n📌 Notes:")
        for note in results["notes"]:
            logger.info(f"  - {note}")
        
        # Save to file
        output_file = Path(f"discovered_configs_{model_short_name}.json")
        existing_configs = {}
        if output_file.exists():
            existing_configs = json.loads(output_file.read_text())
        
        existing_configs.update(config_entry)
        output_file.write_text(json.dumps(existing_configs, indent=4))
        logger.info(f"\n✅ Saved configuration to {output_file}")
        
        logger.info(f"\n=== Discovery Complete ===")
        logger.info(f"Log saved to: {log_file}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()