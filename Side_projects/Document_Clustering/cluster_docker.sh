#!/bin/bash
 
helpFunction()
{
    echo ""
    echo "HELP MANUAL"
    echo "==========="
    echo ""
    echo "This script serves as a setting helper for the containerized development"
    echo "environment of the RnD Topic Modelling Visualization toolkit project."
    echo ""
    echo "Usage:"
    echo "======"
    echo ""
    echo "$0 [-i docker_image] [-n container_name] [-p port]"
    echo ""
    echo -e "\t-i Name for the docker image to be deployed"
    echo -e "\t-n Desired name for the generated docker container"
    echo -e "\t-p Desired port to be forwarded"
    echo ""
    exit 1 # Exit script after printing help
}

# Using Docker in Host
# ---------------------

default_DOCKER_IMAGE="pablorr10/nlp:minimal_mongo"
default_SERVING_PORT=8888

while getopts "i:w:d:n:p:" opt
do
    case "${opt}" in
        i ) DOCKER_IMAGE="${OPTARG}" ;;
        n ) CONTAINER_NAME="${OPTARG}" ;;
        p ) SERVING_PORT="${OPTARG}" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done


if [ -z "${DOCKER_IMAGE}" ]
then
    echo ""
    echo "No docker image has been provide, using default ${default_DOCKER_IMAGE}";
    DOCKER_IMAGE=${default_DOCKER_IMAGE}
fi

if [ -z "${CONTAINER_NAME}" ]
then
    echo "Missing container name: Docker will generate it automatically.";
    CONTAINER_NAME=""
else
    echo "Provided container name: ${CONTAINER_NAME}"
    PARAM_CONTAINER_NAME="--name ${CONTAINER_NAME}"
fi

if [ -z "${SERVING_PORT}" ]
then
    echo "Not specified custom port, using default ${default_SERVING_PORT}"
    ·CLUSTER_PORT=${default_SERVING_PORT}
fi

# Port
INC=10000
DOCKER_PORT=${SERVING_PORT}
CLUSTER_PORT=$(( ${DOCKER_PORT} + ${INC} ))

# DOCKER_IMAGE=pablorr10/nlp:minimal_mongo
# CLUSTER_ROOT=/home/pablo/Side_NLP_Tests/Document_Clustering
# CLUSTER_ROOT=/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering
CLUSTER_ROOT=${PWD}
CONTAINER_ROOT=/app

CLUSTER_DATA=/datadrive/madrid
CONTAINER_DATA=${CONTAINER_ROOT}/globaldata

echo "\n\n"
echo "[INFO] Containerization"
echo "======================="
echo "Docker Image: ${DOCKER_IMAGE}"
echo "Container Name: ${CONTAINER_NAME}"
echo "Docker port: ${DOCKER_PORT}"
echo "Machine port: ${CLUSTER_PORT}"
echo "Docker data: ${CONTAINER_DATA}"
echo "Machine data: ${CLUSTER_DATA}"

# Open a shell in the container
docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true
docker run --rm -dit \
     ${PARAM_CONTAINER_NAME} \
     -e SERVING_PORT=${DOCKER_PORT} \
     -p ${SERVING_PORT}:${DOCKER_PORT} \
     -v ${CLUSTER_ROOT}:${CONTAINER_ROOT} \
     ${DOCKER_IMAGE}

     # -v ${CLUSTER_DATA}:${CONTAINER_DATA} \
     #jupyter notebook --ip='0.0.0.0' --port=8999 --no-browser --allow-root --notebook-dir=${PROJECT_ROOT}
     
echo "[INFO]: Preparing logs..."
sleep 3
docker logs ${CONTAINER_NAME}

echo "[INFO]: Openning shell..."
docker exec -it ${CONTAINER_NAME} bash

echo "[INFO]: Exiting..."
exit

# Usage
sh cluster_docker.sh -n test_nlp -p 8888 



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

