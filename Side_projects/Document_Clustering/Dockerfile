
FROM jupyter/minimal-notebook
# FROM continuumio/miniconda3:latest
# FROM python:3.8
USER root

# sRUN apt-get update && apt-get install sudo
RUN sudo -H apt-get update \
    && apt-get install -y \
        git \
        zip \
        sudo \
        nano \
        htop \
        unzip \
        apt-utils \
    && rm -rf /var/lib/apt/lists/*

ENV IM_IN_DOCKER Yes
ENV PROJECT_ROOT /app
ENV CLUSTER_DATA /data
ENV NLTK_DATA=$PROJECT_ROOT/data/nltk_data  
ENV SERVING_PORT=8889

WORKDIR $PROJECT_ROOT

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN sudo -H rm requirements.txt

EXPOSE 8050
EXPOSE 8888
EXPOSE 8899
EXPOSE 8999
EXPOSE 9999

RUN cat /etc/*-release

RUN pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip
RUN jupyter contrib nbextension install --user
CMD jupyter notebook --ip='0.0.0.0' --port=${SERVING_PORT} --no-browser --allow-root --notebook-dir=${PROJECT_ROOT}
