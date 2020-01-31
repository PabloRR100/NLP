
export CONTAINER_NAME=nlp_mongo_dash
export DOCKER_IMAGE=pablorr10/nlp:mongo_dash

export CONTAINER_ROOT=/app
# export CLUSTER_ROOT=/home/pablo/Side_NLP_Tests/Document_Clustering
export CLUSTER_ROOT=/Users/pabloruizruiz/OneDrive/Courses/NLP_Stanford/Side_projects/Document_Clustering

# Open a shell in the container
docker stop ${CONTAINER_NAME} || true
docker run --rm -dit \
    --name ${CONTAINER_NAME} \
    -e SERVING_PORT=${DOCKER_PORT} \
    -p ${CLUSTER_PORT}:${DOCKER_PORT} \
    -p 5000:5000 \
    -p 8050:8050 \
    -v ${CLUSTER_ROOT}:${CONTAINER_ROOT} \
    -v /Users/pabloruizruiz/Docker/volumes/mongo:/var/lib/mongoql/data \
    ${DOCKER_IMAGE} 

# Rebuild and push image
docker build -t ${DOCKER_IMAGE} ${CLUSTER_ROOT}
docker push ${DOCKER_IMAGE} 

# Start Cluster

# Deploy mongo and dash
kubectl create -f deployment/minikube/mongo
kubectl exec -it <mongo_pod> -- mongo -d app-dev -f home/database/test_db.sql
kubectl exec -it <mongo_pod> -- mongo -U pablominikube -d app-dev --command='SELECT * FROM "<sample_db>"'

kubectl create -f deployment/minikube/dash/

# # Launch Service
minikube service dash-service

# # Apply migration
# kubectl create -f deployment/minikube/dash/job-migration.yaml
# kubectl get pods | grep Error | cut -d' ' -f 1 | xargs kubectl delete pod

# mongo -d app-db-dev -U pablominikube