<a id="vllm--prometheusgrafana"></a>

# vLLM 与 Prometheus／Grafana 监控

[English](README.md)

本示例将 vLLM 指标连接到 Prometheus／Grafana，并通过 Docker 启动两个服务。
其他部署方式见 [Prometheus](https://prometheus.io/) 和 [Grafana](https://grafana.com/) 网站。

安装依赖：

<a id="launch"></a>

### 启动

OpenAI 兼容服务默认启用 Prometheus 指标，通过以下入口启动：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --max-model-len 2048 \
    --disable-log-requests
```

用 `docker compose` 启动 Prometheus 和 Grafana：

```bash
docker compose up
```

向服务发送示例请求：

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 ../../benchmarks/benchmark_serving.py \
    --model mistralai/Mistral-7B-v0.1 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --endpoint /v1/completions \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 3.0
```

访问 [`http://localhost:8000/metrics`](http://localhost:8000/metrics) 查看 vLLM 暴露的原始指标。

<a id="grafana-dashboard"></a>

### Grafana 仪表板

访问 [`http://localhost:3000`](http://localhost:3000)，使用默认用户名和密码 `admin` 登录。

<a id="add-prometheus-data-source"></a>

#### 添加 Prometheus 数据源

访问 [`http://localhost:3000/connections/datasources/new`](http://localhost:3000/connections/datasources/new)，选择 Prometheus。

在配置页的 `Connection` 中填写 `Prometheus Server URL`。本示例两个服务运行在独立容器中，
Docker 为容器提供 DNS 名称，因此可填写 `http://prometheus:9090`。

点击 `Save & Test`，成功时会出现绿色勾及“Successfully queried the Prometheus API.”提示。

<a id="import-dashboard"></a>

#### 导入仪表板

访问 [`http://localhost:3000/dashboard/import`](http://localhost:3000/dashboard/import)，
上传 `grafana.json` 并选择 `prometheus` 数据源，界面如下：

![Grafana Dashboard Image](https://i.imgur.com/R2vH9VW.png)
