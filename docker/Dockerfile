FROM python:3.8-bullseye

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list \
    && apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 \
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
        pandoc \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-plain-generic \
    && rm -rf /var/lib/apt/lists/*

COPY docker/jupyter_nbconvert_config.py /etc/jupyter/jupyter_nbconvert_config.py
COPY docker/jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
RUN mkdir -p /root/.jupyter /usr/local/etc/jupyter \
    && cp /etc/jupyter/jupyter_nbconvert_config.py /root/.jupyter/jupyter_nbconvert_config.py \
    && cp /etc/jupyter/jupyter_nbconvert_config.py /usr/local/etc/jupyter/jupyter_nbconvert_config.py \
    && cp /etc/jupyter/jupyter_server_config.py /root/.jupyter/jupyter_server_config.py \
    && cp /etc/jupyter/jupyter_server_config.py /usr/local/etc/jupyter/jupyter_server_config.py

ENV JUPYTER_CONFIG_PATH=/etc/jupyter:/usr/local/etc/jupyter:/root/.jupyter

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
