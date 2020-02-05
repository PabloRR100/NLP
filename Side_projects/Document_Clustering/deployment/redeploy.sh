

# Destroy previous objects
# kubectl delete pv mongo-pv || true
# kubectl delete pvc mongo-pvc || true 

kubectl delete service dash-service || true 
kubectl delete service mongo-service || true 

kubectl delete deployment dash || true 
kubectl delete deployment mongo || true 


# Deploy mongo and dash
kubectl create -f deployment/minikube/mongo
kubectl wait --for=condition=complete --timeout=120s mongo
kubectl create -f deployment/minikube/dash


# # Launch Service
# minikube service dash-service

# # Apply migration
# kubectl create -f deployment/minikube/dash/job-migration.yaml
