
#!/bin/bash
#######################################################################################
# Author  Jagadeesh Thiruveedula
# Scripting BASH
# OS CENT
# This file will detects the developer code
#and create a docker images as per the code and execute the code inside container!
######################################################################################

figlet MLModel_in_code_and_booting_Docker

if python3 /mlops/codechecker.py == 'CNN'
then
if sudo docker ps -a | grep cnn
then 
sudo docker rm -f cnn
sudo docker run -dit -v /mlops:/mlops --name cnn cnn:v1
else
echo "ML image not found and creating new docker image in runtime"
figlet "Creating CNN ML Image"
docker rmi cnn:v1
mkdir -p /imageml && cd /imageml
docker rmi 
echo -e """
FROM centos \n
RUN yum install python3-pip -y \n
RUN pip3 install --upgrade pip\n
RUN pip3 install tensorflow keras pandas
""" > Dockerfile
docker build -t cnn:v1 .

sleep 5
figlet "Docker Image Built Successfully!"
sudo docker run -dit -v /mlops:/mlops --name cnn cnn:v1
fi
fi

sudo rm -v /mlops/tweak.txt
sudo rm -v /mlops/accuracy.txt
sudo docker exec cnn  python3 /mlops/mnist.py

