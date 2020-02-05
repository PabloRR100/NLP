
FROM jupyter/minimal-notebook 
#:7a0c7325e470
USER root

# RUN apt-get update && apt-get install -yq --no-install-recommends \
RUN sudo apt-get clean
RUN sudo apt-get autoclean
RUN sudo apt-get update && sudo apt-get upgrade
RUN sudo -H apt-get update \
    && apt-get install -y \
        git \
        sudo \
        nano \
        htop \
        apt-utils \
    && rm -rf /var/lib/apt/lists/*

ENV IM_IN_DOCKER Yes
ENV PROJECT_ROOT /app
ENV CLUSTER_DATA /data
ENV NLTK_DATA=$PROJECT_ROOT/data/nltk_data  
ENV SERVING_PORT=8888

WORKDIR $PROJECT_ROOT

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN sudo -H rm requirements.txt

# Mongo
EXPOSE 5000
EXPOSE 8081 
EXPOSE 27017
# Dash
EXPOSE 8050
# Jupyter
EXPOSE 8888
EXPOSE ${SERVING_PORT}

RUN cat /etc/*-release
RUN pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip
RUN jupyter contrib nbextension install --user
CMD jupyter notebook --ip='0.0.0.0' --port=${SERVING_PORT} --no-browser --allow-root --notebook-dir=${PROJECT_ROOT}
