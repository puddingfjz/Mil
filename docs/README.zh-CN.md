<a id="vllm-documents"></a>

# vLLM 文档

[English](README.md)

这里是上游 vLLM 参考文档。Mil 的安装与实验请使用
[artifact 指南](../artifact/README.zh-CN.md)和[环境说明](../artifact/env/README.zh-CN.md)。

<a id="build-the-docs"></a>

## 构建文档

构建文档：安装依赖，清理旧构建并生成 HTML。

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

<a id="open-the-docs-with-your-browser"></a>

## 在浏览器中查看文档

在浏览器中查看文档：启动本地静态文件服务。

```bash
python -m http.server -d build/html/
```

在浏览器中打开 localhost:8000。
