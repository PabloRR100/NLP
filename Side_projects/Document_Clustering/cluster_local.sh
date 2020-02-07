#!/bin/bash
''' Git simulation to syncronize local developement into cluster '''
 
# Preparing environment in Host
# ------------------------------

# SSH into your machine
export SSH_USER=pablo
export SSH_HOST=10.99.195.149

export KEY_PATH="/mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD"
export LOCAL_PROJECT_PATH="/mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/StanfordNLP/Side_projects/Document_Clustering"
export REMOTE_PROJECT_PATH="/home/pablo/Side_NLP_Tests/"
export EXCLUDE_SYNC_FILE="exclude_sync.txt"
 
echo "Usage: $1"

if [ $1 = "push" ]
then
    # Push to remote
    echo "[INFO]: Pusing changes into cluster..."
    rsync -auv -e "ssh -i ${KEY_PATH}" \
    --exclude-from=${EXCLUDE_SYNC_FILE} \
    --delete \
    $LOCAL_PROJECT_PATH ${SSH_USER}@${SSH_HOST}:$REMOTE_PROJECT_PATH
 
elif [ $1 = "port_forwarding" ]
then
    # Open LocalForwarding on the Background
    echo "[INFO]: Opening port forwarding..."
    ssh -i $KEY_PATH \
    -fNL 8050:127.0.0.1:18050 \
    -fNL 8889:127.0.0.1:18889 \
    -fNL 8899:127.0.0.1:18899 \
    -fNL 8999:127.0.0.1:18999 \
    -fNL 9999:127.0.0.1:19999 \
    ${SSH_USER}@${SSH_HOST}

elif [ $1 = "connect" ]
then
    # SSH Connect
    echo "[INFO]: Opening SSH connection..."
    ssh -i ${KEY_PATH} ${SSH_USER}@${SSH_HOST}

elif [ $1 = "retrieve" ]
then
    # Move from Remote to Host --> (Notebooks develped in remote bring to host to add it to Git)
    export REMOTE_DIR="/home/pablo/Side_NLP_Tests/Document_Clustering/Notebooks__Cluster"
    export LOCAL_DIR="/mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/StanfordNLP/Side_projects/Document_Clustering/Notebooks__Cluster"
    echo "Retrieving from $REMOTE_DIR intp $LOCAL_DIR"

    scp -i ${KEY_PATH} \
        -r ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR} ${LOCAL_DIR}
else
    # Print Help
    echo 'The argument passed is not among the possibilites'
fi

exit 

# Usage
sh cluster_local.sh push
sh cluster_local.sh port_forwarding
sh cluster_local.sh connect
sh cluster_local.sh retrieve