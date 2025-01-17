# Installation and Setup Instructions

## Dependencies

### eSpeak-NG

#### macOS

**1. Install Homebrew (if not already installed)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install `espeak-ng`**
```bash
brew install espeak-ng
```

**3. Verify the installation**
```bash
espeak-ng --version
```

#### Windows

**1. Go to `https://github.com/espeak-ng/espeak-ng/releases` and download the \*.msi file**

**2. Execute the downloaded \*.msi file**

**3. Add `C:\Program Files\eSpeak NG\` to your PATH**
   - Search for "Environment Variables" in the Start menu.
   - Click "Edit the system environment variables."
   - In the "System Properties" window, click "Environment Variables."
   - Under "System variables," find and select the Path variable, then click "Edit."
   - Add `C:\Program Files\eSpeak NG\`
   - Click "OK" to save and restart your computer.

**4. Verify the installation**
```bash
espeak-ng --version
```

#### Linux (Debian/Ubuntu-Based Distributions)

**1. Update the package list****
```bash
sudo apt update
```

**2. Install `espeak-ng`**
```bash
sudo apt install espeak-ng
```

**3.Verify the installation**
```bash
espeak-ng --version
```

### FFmpeg

#### macOS

**1. Install Homebrew (if not already installed)**
```bash
/bin/bash -c "\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install ffmpeg**
```bash
brew install ffmpeg
```

**3. Verify the installation**
```bash
ffmpeg -version
```

#### Windows

**1. Go to https://www.gyan.dev/ffmpeg/builds/#release-builds and download the `ffmpeg-release-full.7z` file**

**2. Extract the downloaded file**

**3. Add the the bin folder of the extracted FFmpeg directory to your PATH**
   - Search for "Environment Variables" in the Start menu.
   - Click "Edit the system environment variables."
   - In the "System Properties" window, click "Environment Variables."
   - Under "System variables," find and select the Path variable, then click "Edit."
   - Add the path to the bin folder of the extracted FFmpeg directory (e.g., `C:\ffmpeg\bin`).
   - Click "OK" to save and restart your computer.

**4. Verify the installation**
```bash
ffmpeg -version
```

#### Linux (Debian/Ubuntu-Based Distributions)

```bash
# 1. Update the package list
sudo apt update

# 2. Install ffmpeg
sudo apt install ffmpeg

# 3. Verify the installation
ffmpeg -version
```

### Application

#### macOS and Linux (Debian/Ubuntu-Based Distributions)

**1. Install `uv`**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

**2. Install dependencies**
   ```bash
   uv sync
   ```

#### Windows

**1. Install `uv`**
   Open PowerShell and run
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

**2. Install dependencies**:  
   ```powershell
   uv sync
   ```
---

## Starting the Application

**Run the following command:**  
```bash
uv run app.py
```

---

## Optional Setup: Using Ollama with Docker for Local Models

### macOS, Linux (Debian/Ubuntu-Based Distributions), and Windows

**1. Start the Ollama Docker container**
   ```bash
   docker run --name ollama -d --restart unless-stopped -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
   ```

**2. Pull a completion model, e.g. qwen2.5-coder**
   ```bash
   docker exec ollama ollama pull qwen2.5-coder
   ```

**3. Pull an embedding model e.g. granite-embedding**
   ```bash
   docker exec ollama ollama pull granite-embedding
   ```

---

## Optional Setup: Using SearxNG with Docker for Faster Web Searches

### macOS, Linux (Debian/Ubuntu-Based Distributions), and Windows

**Start the SearxNG Docker container**:  
   ```bash
   docker run --name searxng -d --restart unless-stopped -v $(pwd)/searxng:/etc/searxng:rw -e UWSGI_WORKERS=${SEARXNG_UWSGI_WORKERS:-4} -e UWSGI_THREADS=${SEARXNG_UWSGI_THREADS:-4} -p 8080:8080 docker.io/searxng/searxng:latest
   ```

---

## Troubleshooting

### ImportError libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12

#### macOS and Linux (Debian/Ubuntu-Based Distributions)

1. **Activate the virtual environment**
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
