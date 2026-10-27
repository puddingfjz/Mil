ARG BASE_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel@sha256:a5de097b482f5927baf2322f4419f11044bfe4f08c7b7593dbaff8e41d03a204
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    TORCH_CUDA_ARCH_LIST=8.0

RUN conda create --yes --name mil-artifact \
        python=3.9.20 pip=24.2 setuptools=75.1.0 wheel=0.44.0 \
    && conda clean --all --yes

COPY artifact/env/runtime-freeze.txt /tmp/mil-runtime-freeze.txt
RUN conda run --no-capture-output -n mil-artifact \
    python -m pip install --no-cache-dir -r /tmp/mil-runtime-freeze.txt

WORKDIR /workspace/vllm
COPY . .
RUN conda run --no-capture-output -n mil-artifact \
    python -m artifact.mil_artifact.runtime_identity build
RUN conda run --no-capture-output -n mil-artifact \
    python -m artifact.mil_artifact.runtime_identity verify
RUN conda run --no-capture-output -n mil-artifact \
    python -m unittest discover -s artifact/tests -v

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mil-artifact", "python", "-m", "artifact.mil_artifact.run"]
CMD ["--profile", "smoke", "--dry-run"]
