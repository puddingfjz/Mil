<a id="contributing-to-vllm"></a>

# 为 vLLM 贡献

[English](CONTRIBUTING.md)

欢迎参与 vLLM 社区，无论贡献大小。可以报告问题、请求或添加模型，以及提出或实现新功能。

贡献不限于代码，回答问题、帮助其他使用者和改进文档同样有价值。

也可以通过博客介绍使用 vLLM 的项目、在 Twitter 分享使用体验，或为仓库加星来支持项目。

<a id="setup-for-development"></a>

## 开发环境设置

<a id="build-from-source"></a>

### 从源码构建

从源码构建；安装可能需要几分钟：

```bash
pip install -r requirements.txt
pip install -e .  # This may take several minutes.
```

<a id="testing"></a>

### 测试

测试：先安装开发依赖，再运行静态类型检查和单元测试。

```bash
pip install -r requirements-dev.txt

# Static type checking
mypy
# Unit tests
pytest tests/
```

**注意：** 当前仓库不能通过全部 mypy 检查。

<a id="contributing-guidelines"></a>

## 贡献指南

<a id="issue-reporting"></a>

### 报告问题

遇到错误或有功能请求时，先搜索已有 issue。若尚未有人报告，创建新 issue 并提供尽可能完整的相关信息。

<a id="coding-style-guide"></a>

### 代码风格

代码遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)和
[Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)，可使用
[`format.sh`](./format.sh) 格式化。

<a id="pull-requests"></a>

### 提交合并请求

提交合并请求时：

1. 将修改变基到 main 分支的最新提交。
2. 运行 [`format.sh`](./format.sh) 确保格式正确。
3. 详细说明修改内容及原因；若修复已有 issue，在描述中引用它。

<a id="code-reviews"></a>

### 代码审阅

所有提交（包括项目成员的提交）均需代码审阅。请保持修改简洁；涉及互不相关的内容时，
考虑拆成多个合并请求。及时回应审阅意见；对不清楚或有分歧的建议，可进一步询问和讨论。

<a id="thank-you"></a>

### 致谢

感谢阅读贡献指南并参与 vLLM，社区贡献让项目持续改进。
