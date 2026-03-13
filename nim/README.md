# Nemotron-Nano-9B NIM Container

## Quick Start

```bash
# Login to NGC (done during Phase 0)
source .env
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Start NIM container via Docker Compose
docker compose up nemotron-nim

# Or standalone:
docker run --rm --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v ./nim-cache:/opt/nim/.cache \
  -p 8001:8000 \
  --shm-size=16g \
  nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:latest
```

## API

The NIM container exposes an OpenAI-compatible API at port 8001:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark",
    "messages": [{"role": "user", "content": "What is price elasticity?"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Integration

The backend connects to NIM via `services/llm_client.py`. Set `NIM_URL` in `.env` or
Docker Compose to point to the NIM container (default: `http://localhost:8001`).

## DGX Spark Notes

- GB10 with 128GB unified memory can run the 9B parameter model comfortably
- First load takes ~2-5 minutes to download and cache the model
- Subsequent starts use the cache at `./nim-cache/`
