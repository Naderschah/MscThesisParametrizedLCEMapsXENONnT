# FROM tensorflow/tensorflow:2.17.0-gpu-jupyter 
# Ubuntu 22.04
# Pyton 3.11
# tensorflow 2.17.0
# CUDA 12.3.0 


FROM opensciencegrid/osgvo-xenon:el9.2024.10.4
# CudNN version mismatch with tf and Cuda see https://www.tensorflow.org/install/source#gpu (cudnn 8.9 vs 8.4 -> How does this work on server?)
RUN /bin/bash -c "source /opt/XENONnT/anaconda/bin/activate && conda activate XENONnT_el9.2024.10.4 && pip install pycharge" # Version used in singularity image -> Just runnign this to update deps:
#RUN apt update && apt install -y git wget vim

# Need tfp and it turns out manually copying out source is too much work

#RUN pip install numpy scipy
#RUN pip install numba
#RUN pip install matplotlib
#RUN pip install straxen==2.2.4
# RUN pip install jupyter
ENV XENON_CONFIG="/Code/xenon.config"
