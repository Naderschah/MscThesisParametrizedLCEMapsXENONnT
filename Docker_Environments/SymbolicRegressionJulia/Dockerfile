#FROM jupyter/datascience-notebook:x86_64-ubuntu-22.04
FROM julia:1.10.7-bookworm

USER root
# Install required system dependencies for HDF5 and Jupyter
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    hdf5-tools \
    git \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir jupyterlab notebook pandas tensorflow tensorboard matplotlib 
RUN pip3 install --break-system-packages --no-cache-dir straxen
# Install Julia packages - For some reason it installs a really old SymbollicRegression
RUN julia -e "using Pkg; \
    Pkg.add([\"IJulia\", \"MLJBase\", \"PyCall\", \"SymbolicRegression\", \"HDF5\", \"DataFrames\", \"LoopVectorization\", \"ProgressMeter\", \"SymbolicUtils\", \"PyPlot\", \"TensorBoardLogger\", \"Latexify\", \"ClusterManagers\"]); \  
    Pkg.precompile()"

ENV JULIA_NUM_THREADS=20
