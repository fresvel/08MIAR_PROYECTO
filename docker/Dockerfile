FROM python:3.8-bullseye

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglu1-mesa \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        xvfb \
        xauth \
        xfonts-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
