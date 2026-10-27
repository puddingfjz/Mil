# vLLM documents

[简体中文](README.zh-CN.md)

These are upstream vLLM references. For Mil, use the [artifact guide](../artifact/README.md)
and [installation instructions](../artifact/env/README.md).

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d build/html/
```

Launch your browser and open localhost:8000.
