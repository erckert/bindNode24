# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN adduser -u 5678 --shell /bin/bash --disabled-password --gecos "" appuser && chown -R appuser /app
# Install pip requirements
RUN pip install --no-cache-dir -r requirements.txt && apt update && apt install -y git && pip install --no-cache-dir --upgrade git+https://github.com/huggingface/diffusers.git

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers

USER appuser
