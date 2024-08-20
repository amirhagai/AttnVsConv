# Use the PyTorch 2.0.1 image as the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean

# Install the necessary Python packages
RUN pip install --upgrade pip && \
    pip install \
    numpy \
    einops \
    torchvision \
    pandas \
    scikit-learn \
    opencv-python-headless \
    pillow \
    matplotlib \
    seaborn && \
    pip install --upgrade diffusers[torch] && \
    pip install transformers && \
    pip install datasets && \
    pip install peft && \
    pip install accelerate && \
    python -m pip install optimum && \
    pip install git+https://github.com/RobustBench/robustbench.git

# Copy your application code into the container
# (Assuming your application code is in the same directory as this Dockerfile)
#COPY . /app

