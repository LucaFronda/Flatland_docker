# Flatland_docker
Docker for running the flatland software

# 1- First install docker on your pc:

curl -s-Lhttps://nvidia.github.io/nvidia-container-runtime/gpgkey | \
sudo apt-key add–distribution=$(./etc/os-release;echo$ID$VERSION_ID)
curl -s-Lhttps://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
sudo tee/etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt install nvidia-container-runtime
sudo systemctl restart daemon-reload
sudo systemctl  restart docker

p.s. Be sure to have CUDA working on your machine!

# 2- Define the dockerfile and the requirements.txt

Choose the right cuda version...

# 3- Build the docker-image

docker build -t choose-an-image-name .

Check existing images on the machine:
docker images

Check running containers:
docker container ls

Delete existing image:
docker rmi –f your-image-id (id is visible with the docker images command)

# 4- Run the docker and control if your python script work

Interactively:
docker run -it --gpus all your-image-id /bin/bash

Single command:
docker run -–gpus all your-image-id python your-script.py

# 5- If all works push the docker image on docker hub

docker login --username=your-username --password=your-pw 
docker tag your-image-id your-username/your-image-name:new-tag
docker push your-username/your-image-name

# 6- Pull the image from docker hub to the DGX:

cd $CINECA_SCRATCH  (this if your image is big, probable thing)
singularity pull docker://your-docker-username/your-image-name:tag

# 7- Run the docker on the DGX:
srun -N1 --gres=gpu:2 -A IscrC_DSD_0  --time=00:02:00 --pty bash (this has to be changed depending on the job resources needed)
singularity exec --nv your-image-name.sif python your-example.py

# For more documentation on the command for the DGX check https://hpc.llnl.gov/banks-jobs/running-jobs/slurm-commands



