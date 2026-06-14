# Pull the StoryForge generation model into the Ollama Docker container.
param(
    [string]$Model = "qwen3.5:9b"
)

$ErrorActionPreference = "Stop"

Write-Host "Pulling Ollama model: $Model"
docker compose exec ollama ollama pull $Model
Write-Host "Done. Models in container:"
docker compose exec ollama ollama list
