#!/usr/bin/env python3
"""
Setup script for Ollama integration.

This script helps set up the Ollama environment and pulls recommended models.
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description if description else cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_ollama_installed():
    """Check if Ollama is installed locally or its Docker image exists."""
    try:
        # Check local binary
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama is installed locally: {result.stdout.strip()}")
            return True
        else:
            print("✗ Ollama is not installed or not in PATH")
    except FileNotFoundError:
        print("✗ Ollama is not installed locally")

    # Check if Docker image exists
    try:
        docker_images = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True
        )
        if docker_images.returncode == 0:
            images = docker_images.stdout.strip().splitlines()
            matches = [img for img in images if "ollama" in img.lower()]
            if matches:
                print("✓ Ollama Docker image found:")
                for m in matches:
                    print(f"  - {m}")
                return True
            else:
                print("✗ No Ollama Docker image found")
        else:
            print("✗ Docker command failed")
    except FileNotFoundError:
        print("✗ Docker is not installed or not in PATH")

    return False


def check_ollama_running():
    """Check if Ollama service is running natively or as a Docker container."""
    # Check native service
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama service is running natively")
            return True
        else:
            print("✗ Ollama service is not running natively")
    except FileNotFoundError:
        print("✗ Ollama CLI not found")

    # Check Docker container
    try:
        docker_ps = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}} {{.Image}} {{.Names}}"],
            capture_output=True, text=True
        )
        if docker_ps.returncode == 0:
            containers = docker_ps.stdout.strip().splitlines()
            matches = [c for c in containers if "ollama" in c.lower()]
            if matches:
                print("✓ Ollama service is running in Docker:")
                for m in matches:
                    print(f"  - {m}")
                return True
            else:
                print("✗ No running Ollama Docker container found")
        else:
            print("✗ Docker command failed")
    except FileNotFoundError:
        print("✗ Docker not installed or not in PATH")

    return False


def list_available_models():
    """List currently available models from native Ollama and/or Docker Ollama."""
    models_found = False
    all_outputs = []

    # Native check
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print("📦 Native Ollama models:")
            print(result.stdout.strip())
            all_outputs.append(result.stdout.strip())
            models_found = True
        else:
            print("✗ No models found in native Ollama")
    except FileNotFoundError:
        print("✗ Ollama CLI not found")

    # Docker check
    try:
        docker_ps = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}} {{.Image}} {{.Names}}"],
            capture_output=True, text=True
        )
        containers = [c.split()[0] for c in docker_ps.stdout.strip().splitlines() if "ollama" in c.lower()]
        if containers:
            for cid in containers:
                docker_list = subprocess.run(
                    ["docker", "exec", cid, "ollama", "list"],
                    capture_output=True, text=True
                )
                if docker_list.returncode == 0 and docker_list.stdout.strip():
                    print(f"📦 Ollama models in Docker container {cid}:")
                    print(docker_list.stdout.strip())
                    all_outputs.append(docker_list.stdout.strip())
                    models_found = True
                else:
                    print(f"✗ Could not list models in Docker container {cid}")
        else:
            print("✗ No Ollama Docker container found")
    except FileNotFoundError:
        print("✗ Docker not installed or not in PATH")

    if not models_found:
        print("⚠ No available models found in native or Docker Ollama")
        return ""
    
    return "\n\n".join(all_outputs)


def create_secrets_file():
    """Create a minimal SECRETS file for Ollama usage."""
    secrets_path = Path("SECRETS")
    
    if secrets_path.exists():
        print("✓ SECRETS file already exists")
        return True
    
    print("Creating minimal SECRETS file for Ollama usage...")
    secrets_content = """API_KEY=placeholder
LLAMA_API_BASE=http://localhost:11434
NYU_ORG=placeholder
ARG_ORG=placeholder
ANTHROPIC_API_KEY=placeholder
MISTRAL_API_KEY=placeholder
REPLICATE_API_KEY=placeholder
"""
    
    try:
        with open(secrets_path, 'w') as f:
            f.write(secrets_content)
        print("✓ Created SECRETS file")
        return True
    except Exception as e:
        print(f"✗ Failed to create SECRETS file: {e}")
        return False


def pull_model(model_name, target="native", container_name=None):
    """Pull a specific model for native Ollama or Docker container."""
    if target == "native":
        return run_command(f"ollama pull {model_name}", f"Pulling model: {model_name} (native)")

    elif target == "docker":
        if not container_name:
            print("✗ No Docker container specified for Ollama.")
            return False
        return run_command(
            f"docker exec {container_name} ollama pull {model_name}",
            f"Pulling model: {model_name} (Docker: {container_name})"
        )

    else:
        print("Invalid target. Must be 'native' or 'docker'.")
        return False


def main():
    print("Ollama Setup Script for Unsupervised-Elicitation")
    print("=" * 60)
    
    # Create SECRETS file first
    create_secrets_file()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\nPlease install Ollama first:")
        print("- Visit: https://ollama.ai/")
        print("- Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return
    
     # Ask target type before pulling
    target_choice = ""
    while target_choice not in ["native", "docker"]:
        target_choice = input("\nPull models for 'native' Ollama app or 'docker' container? [native/docker]: ").strip().lower()
        if target_choice not in ["native", "docker"]:
            print("Please enter 'native' or 'docker'.")

    # If Docker selected, check running containers
    container_name = None
    if target_choice == "docker":
        docker_ps = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}} {{.Image}}"],
            capture_output=True, text=True
        )
        containers = [line.split()[0] for line in docker_ps.stdout.strip().splitlines() if "ollama" in line.lower()]
        if not containers:
            print("✗ No running Ollama Docker container found. Please start one first.")
            return
        elif len(containers) > 1:
            print("\nMultiple Ollama containers detected:")
            for idx, name in enumerate(containers, start=1):
                print(f"{idx}. {name}")
            choice = input("Select container number: ").strip()
            try:
                container_name = containers[int(choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice.")
                return
        else:
            container_name = containers[0]
            print(f"Using detected container: {container_name}")

    # Check if Ollama is running
    if not check_ollama_running():
        print("\nStarting Ollama service...")
        print("Run this in a separate terminal: ollama serve")
        input("Press Enter after starting Ollama service...")
        
        if not check_ollama_running():
            print("Ollama service is still not running. Please start it manually.")
            return
    
    # List current models
    print("\n" + "="*60)
    print("CURRENT MODELS")
    print("="*60)
    list_available_models()
    
    # Recommended models for this project
    recommended_models = [
        ("llama3.2:latest", "Latest Llama 3.2 model - good general purpose"),
        ("llama3.1:8b", "Llama 3.1 8B - balanced performance/speed"),
        ("codellama:latest", "Code generation specialist"),
        ("mistral:latest", "Fast and efficient general model"),
        ("qwen2.5:7b", "Good multilingual capabilities"),
    ]
    
    print("\n" + "="*60)
    print("RECOMMENDED MODELS")
    print("="*60)
    
    for model, description in recommended_models:
        while True:
            response = input(f"\nPull {model}? ({description}) [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                success = pull_model(model, target_choice, container_name)
                if success:
                    print(f"✓ Successfully pulled {model}")
                else:
                    print(f"✗ Failed to pull {model}")
                break
            elif response in ['n', 'no', '']:
                print(f"Skipping {model}")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    print("\nWould you like to pull some recommended models?")
    print("Note: Models can be large (several GB each)")
    
    # Ask user which models to pull
    for model, description in recommended_models:
        while True:
            response = input(f"\nPull {model}? ({description}) [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                success = pull_model(model)
                if success:
                    print(f"✓ Successfully pulled {model}")
                else:
                    print(f"✗ Failed to pull {model}")
                break
            elif response in ['n', 'no', '']:
                print(f"Skipping {model}")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    # Final status
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)
    list_available_models()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run the verification script:")
    print("   python verify_ollama_integration.py")
    print()
    print("3. Run the example script:")
    print("   python examples/ollama_example.py")
    print()
    print("4. Or use in your code:")
    print("   from core.llm_api import ModelAPI")
    print("   model_api = ModelAPI()")
    print("   response = await model_api.call_single('llama3.2:latest', 'Hello!', max_tokens=100)")
    print()
    print("5. Browse more models at: https://ollama.ai/library")


if __name__ == "__main__":
    main()
