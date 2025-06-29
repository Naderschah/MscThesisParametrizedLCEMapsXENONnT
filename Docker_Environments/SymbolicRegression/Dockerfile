FROM jupyter/base-notebook:x86_64-ubuntu-22.04

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
USER root
# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    git \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install core scientific Python packages and Jupyter Notebook
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyter \
    tensorflow \
    notebook

# Install additional Python packages
RUN pip install --no-cache-dir \
    straxen \
    pysr

ENV XENON_CONFIG="/Code/xenon.config"
