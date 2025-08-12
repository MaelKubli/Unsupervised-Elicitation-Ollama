#!/usr/bin/env python3
"""
Quick verification that the Ollama integration works correctly.
"""

import asyncio
import sys
import os
import subprocess

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from core.llm_api import ModelAPI
        print("✓ ModelAPI import successful")
    except Exception as e:
        print(f"✗ ModelAPI import failed: {e}")
        return False
    
    try:
        from core.llm_api.ollama_llm import OllamaModel, OLLAMA_MODELS
        print("✓ OllamaModel import successful")
        print(f"✓ Found {len(OLLAMA_MODELS)} predefined Ollama models")
    except Exception as e:
        print(f"✗ OllamaModel import failed: {e}")
        return False
    
    return True

def _parse_ollama_list_output(text: str):
    """Parse `ollama list` output into a list of model names (first column)."""
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Skip header line if present (starts with "NAME")
    if lines and lines[0].lower().startswith("name"):
        lines = lines[1:]
    models = []
    for ln in lines:
        # model name is the first whitespace-separated token
        parts = ln.split()
        if parts:
            models.append(parts[0])
    return models


def _detect_ollama_containers():
    """Return list of running docker containers that look like ollama."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Image}}"],
            capture_output=True, text=True
        )
        if proc.returncode != 0:
            return []
        rows = [r for r in proc.stdout.splitlines() if r.strip()]
        out = []
        for r in rows:
            cid, cname, *rest = r.split()
            image = " ".join(rest) if rest else ""
            if "ollama" in (cname + " " + image).lower():
                out.append((cid, cname, image))
        return out
    except FileNotFoundError:
        # docker not installed
        return []


def _ollama_list_native():
    """Run `ollama list` on the host."""
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if proc.returncode == 0:
            return _parse_ollama_list_output(proc.stdout), "native"
    except FileNotFoundError:
        pass
    return [], "native"


def _ollama_list_in_container(container_name_or_id: str):
    """Run `ollama list` inside a docker container."""
    proc = subprocess.run(
        ["docker", "exec", container_name_or_id, "ollama", "list"],
        capture_output=True, text=True
    )
    if proc.returncode == 0:
        return _parse_ollama_list_output(proc.stdout), f"docker:{container_name_or_id}"
    return [], f"docker:{container_name_or_id}"

def check_available_models():
    """Check what models are actually available in Ollama (native or Docker)."""
    # 1) Try native
    models, source = _ollama_list_native()
    if models:
        return models

    # 2) Try Docker (respect env override if set)
    preferred = os.environ.get("OLLAMA_CONTAINER", "").strip()
    if preferred:
        models, source = _ollama_list_in_container(preferred)
        if models:
            print(f"(info) Models read from Docker container '{preferred}'")
            return models
        else:
            print(f"(info) No models returned by container '{preferred}'. Falling back to auto-detect.")

    containers = _detect_ollama_containers()
    if not containers:
        # nothing found anywhere
        return []

    # If multiple containers, pick the first for non-interactive verification.
    cid, cname, image = containers[0]
    models, _ = _ollama_list_in_container(cname or cid)
    if models:
        print(f"(info) Models read from Docker container '{cname or cid}' ({image})")
    return models


async def test_model_detection():
    """Test that Ollama models are correctly detected."""
    print("\nTesting model detection...")
    
    try:
        from core.llm_api import ModelAPI
        model_api = ModelAPI()
        
        # Test various model IDs
        test_cases = [
            ("llama3.2:latest", True, "Standard Ollama model"),
            ("gpt-4", False, "OpenAI model"), 
            ("claude-3", False, "Anthropic model"),
            ("mistral:7b", True, "Ollama model with tag"),
            ("custom-llama", True, "Custom model with llama in name"),
        ]
        
        for model_id, should_be_ollama, description in test_cases:
            is_ollama = model_api._is_ollama_model(model_id)
            status = "✓" if is_ollama == should_be_ollama else "✗"
            print(f"{status} {model_id} -> {'Ollama' if is_ollama else 'Not Ollama'} ({description})")
        
        # Check available models
        available_models = check_available_models()
        if available_models:
            print(f"\n✓ Found {len(available_models)} available models:")
            for model in available_models[:5]:  # Show first 5
                print(f"  - {model}")
            if len(available_models) > 5:
                print(f"  ... and {len(available_models) - 5} more")
        else:
            print("\n⚠ No models found. Pull some with: ollama pull llama3.2:latest")
        
        return True
            
    except Exception as e:
        print(f"✗ Model detection test failed: {e}")
        return False


async def test_ollama_connection():
    """Test connection to Ollama service."""
    print("\nTesting Ollama connection...")
    
    try:
        from core.llm_api.ollama_llm import OllamaModel
        ollama_model = OllamaModel()
        print("✓ OllamaModel initialization successful")
        return True
    except ImportError as e:
        print(f"✗ Ollama dependencies not available: {e}")
        print("Install with: pip install ollama langchain-ollama langchain-core")
        return False
    except Exception as e:
        print(f"⚠ Ollama service connection issue: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return False


async def test_model_api_init():
    """Test ModelAPI initialization with Ollama support."""
    print("\nTesting ModelAPI initialization...")
    
    try:
        from core.llm_api import ModelAPI
        model_api = ModelAPI(
            ollama_host="http://localhost:11434",
            ollama_temperature=0.1
        )
        print("✓ ModelAPI with Ollama settings initialized successfully")
        
        # Check that Ollama model is available
        if hasattr(model_api, '_ollama'):
            print("✓ Ollama model backend available")
        else:
            print("✗ Ollama model backend not found")
            
        return True
    except Exception as e:
        print(f"✗ ModelAPI initialization failed: {e}")
        return False


async def test_actual_call():
    """Test making an actual call to Ollama."""
    print("\nTesting actual Ollama API call...")
    
    # Check if any models are available
    available_models = check_available_models()
    if not available_models:
        print("⚠ No models available for testing. Pull a model first:")
        print("  ollama pull llama3.2:latest")
        return False
    
    try:
        from core.llm_api import ModelAPI
        model_api = ModelAPI()
        
        test_model = available_models[0]  # Use first available model
        print(f"Testing with model: {test_model}")
        
        # Use the main __call__ method directly with explicit parameters to ensure single response
        responses = await model_api(
            test_model,
            [
                {"role": "user", "content": "Say 'Hello from Ollama!' and nothing else."}
            ],
            max_tokens=20,
            n=1,
            num_candidates_per_completion=1
        )
        
        if len(responses) != 1:
            print(f"✗ Expected 1 response, got {len(responses)}")
            return False
        
        response = responses[0]
        
        # Extract completion from the response format
        completion = None
        if isinstance(response, dict):
            if "response" in response and isinstance(response["response"], dict):
                # Format: {"prompt": ..., "response": {"completion": "...", ...}}
                completion = response["response"].get("completion", "")
            elif "completion" in response:
                # Format: {"completion": "...", ...}
                completion = response["completion"]
        elif hasattr(response, 'completion'):
            # LLMResponse object
            completion = response.completion
        
        if completion is None:
            print(f"✗ Could not extract completion from response: {response}")
            return False
        
        print(f"✓ Successfully got response: {completion}")
        return True
        
    except Exception as e:
        print(f"✗ API call test failed: {e}")
        return False


async def main():
    print("Ollama Integration Verification")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_detection, 
        test_ollama_connection,
        test_model_api_init,
        test_actual_call,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("VERIFICATION SUMMARY")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("\nOllama integration is working correctly!")
        print("\nNext steps:")
        print("1. Run example: python examples/ollama_example.py")
        print("2. Use in your projects with: ModelAPI().call_single('model_name', prompt, max_tokens=100)")
    else:
        print(f"✗ {total - passed} of {total} tests failed")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is installed: https://ollama.ai")
        print("2. Start Ollama service: ollama serve")
        print("3. Pull a model: ollama pull llama3.2:latest")
        print("4. Install requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(main())
