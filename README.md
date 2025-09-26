# Install UV (if not already installed)
## Windows
```PS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
## Mac and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Create new project
```bash
uv init rl-chatbot
cd rl-chatbot
```

# Add dependencies
```bash
uv add gymnasium torch transformers 
```

```bash
sentence-transformers faiss-cpu numpy pandas
```

# How to run
```bash
uv run main.py
```
