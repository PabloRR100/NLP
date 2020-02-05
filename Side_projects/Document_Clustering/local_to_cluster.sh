
# Preparing environment in Host
# ------------------------------

# SSH into your machine
export SSH_USER=pablo
export SSH_HOST=10.99.195.149

export KEY_PATH="/mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD"
export LOCAL_PROJECT_PATH="/mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/StanfordNLP/Side_projects/Document_Clustering"
export REMOTE_PROJECT_PATH="/home/pablo/Side_NLP_Tests/"
export EXCLUDE_SYNC_FILE="exclude_sync.txt"

# Push to remote
rsync -auv -e "ssh -i ${KEY_PATH}" \
    --exclude-from=${EXCLUDE_SYNC_FILE} \
    $LOCAL_PROJECT_PATH ${SSH_USER}@${SSH_HOST}:$REMOTE_PROJECT_PATH

# SSH Connect
ssh -i ${KEY_PATH} ${SSH_USER}@${SSH_HOST}

# Open LocalForwarding on the Background
ssh -i $KEY_PATH \
    -fNL 8889:127.0.0.1:18889 \
    -fNL 8899:127.0.0.1:18899 \
    -fNL 8999:127.0.0.1:18999 \
    -fNL 9999:127.0.0.1:19999 \
    pablo@10.99.195.149

    
# Using Docker in Host
# ---------------------

export DOCKER_PORT=8999
export CLUSTER_PORT=18999
export DOCKER_PORT_2=8050
export CLUSTER_PORT_2=8050

export DOCKER_IMAGE=pablorr10/nlp:minimal_mongo
export CONTAINER_NAME=nlp_minimal_mongo

# export CLUSTER_ROOT=/home/pablo/Side_NLP_Tests/Document_Clustering
export CLUSTER_ROOT=/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering
export CONTAINER_ROOT=/app

export CLUSTER_DATA=/datadrive/madrid
export CONTAINER_DATA=${CONTAINER_ROOT}/globaldata

# Open a shell in the container
docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true
docker run --rm -dit \
    --name ${CONTAINER_NAME} \
    -e SERVING_PORT=${DOCKER_PORT} \
    -p ${CLUSTER_PORT}:${DOCKER_PORT} \
    -v ${PWD}:${CONTAINER_ROOT} \
    ${DOCKER_IMAGE} jupyter notebook --ip='0.0.0.0' --port=8999 --no-browser --allow-root --notebook-dir=${PROJECT_ROOT}
    # -p ${CLUSTER_PORT_2}:${DOCKER_PORT_2} \
    # -v ${CLUSTER_DATA}:${CONTAINER_DATA} \
    
    jupyter contrib nbextension install --user # Rebuild image

docker logs ${CONTAINER_NAME}

docker exec -it ${CONTAINER_NAME} bash

# Move from Remote to Host --> (Notebooks develped in remote bring to host to add it to Git)
export REMOTE_DIR="/home/pablo/Side_NLP_Tests/Document_Clustering/Notebooks__Cluster"
export LOCAL_DIR="/mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/StanfordNLP/Side_projects/Document_Clustering/Notebooks__Cluster"

scp -i ${KEY_PATH} \
    -r ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR} ${LOCAL_DIR}


# On BASF Laptop
# --------------
docker stop nlp_test || true
docker run --rm -dit --name nlp_test -e SERVING_PORT=8899 -p 8899:8899 pablorr10/nlp:minimal 
# jupyter notebook --ip='0.0.0.0' --port=8899 --no-browser --allow-root --notebook-dir=/app


# On my MAC
# ---------

export DOCKER_IMAGE=pablorr10/nlp:minimal
export CONTAINER_NAME=nlpdash
export CONTAINER_ROOT=/app

docker run --rm -dit \
    --name ${CONTAINER_NAME} \
    -e SERVING_PORT=8899 \
    -p 8899:8899 \
    -v ${PWD}:${CONTAINER_ROOT} \
    ${DOCKER_IMAGE} jupyter notebook --ip='0.0.0.0' --port=8899 --no-browser --allow-root --notebook-dir=/app

docker logs ${CONTAINER_NAME}

docker exec -dit ${CONTAINER_NAME} python frontend/cluster_frontend.py
docker logs ${CONTAINER_NAME}

docker exec -it ${CONTAINER_NAME} bash

