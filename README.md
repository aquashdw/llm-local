# LLM Local

playing around

```bash
# for macos, after uv sync run:
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 uv pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

grpc code generation

```bash
uv run -m grpc_tools.protoc -I./protos --python_out=. --pyi_out=. --grpc_python_out=. protos/chatting.proto
```

