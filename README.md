# Installation and Setup Instructions

## Dependencies

### macOS

**Install Homebrew (if not already installed)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install `espeak-ng`**
```bash
brew install espeak-ng
```

**Verify the installation**
```bash
espeak-ng --version
```


## General Instructions

### macOS and Linux

1. **Install `uv`**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

### Windows

1. **Install `uv`**
   Open PowerShell and run
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install dependencies**:  
   ```powershell
   uv sync
   ```
---

## Optional Setup: Using Ollama for Local Models

### macOS, Linux, and Windows (with Docker)

1. **Start the Ollama Docker container**:  
   ```bash
   docker run --name ollama -d --restart unless-stopped -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
   ```

2. **Pull a completion model, e.g. qwen2.5-coder**:  
   ```bash
   docker exec ollama ollama pull qwen2.5-coder
   ```

3. **Pull an embedding model e.g. granite-embedding**:  
   ```bash
   docker exec ollama ollama pull granite-embedding
   ```

---

## Optional Setup: Using SearxNG for Faster Web Searches

### macOS, Linux, and Windows (with Docker)

**Start the SearxNG Docker container**:  
   ```bash
   docker run --name searxng -d --restart unless-stopped -v $(pwd)/searxng:/etc/searxng:rw -e UWSGI_WORKERS=${SEARXNG_UWSGI_WORKERS:-4} -e UWSGI_THREADS=${SEARXNG_UWSGI_THREADS:-4} -p 8080:8080 docker.io/searxng/searxng:latest
   ```

---

## Starting the Application

**Run the following command:**  
```bash
uv run app.py
```

---

## Troubleshooting

### Issue: `ImportError: [...]/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`

#### macOS and Linux

1. Activate the virtual environment:  
   ```bash
   source .venv/bin/activate
   ```

2. Set the `LD_LIBRARY_PATH`:  
   ```bash
   export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
   ```

3. Start the application:  
   ```bash
   python3 app.py
   ```

#### Windows

1. Ensure the correct environment is activated in your terminal.

2. Adjust the library path in a similar manner using your Python site-packages path.

3. Start the application:  
   ```powershell
   python app.py
   ```

---
