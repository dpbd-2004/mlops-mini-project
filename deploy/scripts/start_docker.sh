#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 361753377361.dkr.ecr.eu-north-1.amazonaws.com

# Pull the latest image
docker pull 361753377361.dkr.ecr.eu-north-1.amazonaws.com/dp_ecr:latest

# Check if the container 'campusx-app' is running
if [ "$(docker ps -q -f name=mlops-app)" ]; then
    # Stop the running container
    docker stop mlops-app
fi

# Check if the container 'campusx-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=mlops-app)" ]; then
    # Remove the container if it exists
    docker rm mlops-app
fi

# Run a new container
docker run -d -p 80:5000 -e DAGSHUB_PAT=8739f6c351adb871d5d080540aa06026e504b1b6 --name mlops-app 361753377361.dkr.ecr.eu-north-1.amazonaws.com/dp_ecr:latest