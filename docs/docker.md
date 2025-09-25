### Docker Setup

All of the examples in this repo provide instructions for being run normally, and also using Docker. Although not required, the Docker option is recommended as this will simplify software installation, produce a more stable environment, and also allow you to avoid installing ROS and cluttering your machine, for examples which depend on ROS.

- Basic Docker installation instructions are [here](https://docs.docker.com/engine/install/).
- Docker must be installed in [rootless mode](https://docs.docker.com/engine/security/rootless/).
- To use your GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The version of docker installed with `snap` is incompatible with the NVIDIA container toolkit, preventing it from accessing `libnvidia-ml.so` ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)). The snap version can be uninstalled with `sudo snap remove docker`.
- Docker Desktop is also incompatible with the NVIDIA runtime ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)). Docker Desktop can be uninstalled with `sudo apt remove docker-desktop`.


If starting from scratch and your host machine is Ubuntu 22.04, you can use accomplish all of the above with the convenience scripts `scripts/docker/install_docker_ubuntu22.sh` and `scripts/docker/install_nvidia_container_toolkit.sh`.

Build the Docker image and start the container with the following command:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

To build and run the Docker image for a specific example, use the following command:
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
where `<example_name>` is the name of the example you want to run.

During the first run of any example, Docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

### Vast.ai Docker Workflow

We provide a ready-to-use Docker setup at the repository root. This container bundles the repository and dependencies so you do not need to clone the repo or run `uv sync` manually on remote machines.

1. **Build the image (optional)**
   ```bash
   docker build -t openpi-ur5e:latest .
   ```
   Publish the image to your container registry if you want to pull it from vast.ai instances.

2. **Run the container**
   ```bash
   docker run --gpus all -it \
     -e WANDB_API_KEY=your_wandb_key \
     -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
     -v /path/to/persistent/checkpoints:/workspace/checkpoints \
     -v /path/to/persistent/assets:/workspace/assets \
     openpi-ur5e:latest
   ```

   Inside the container use the helper script:
   ```bash
   ./setup_and_train.sh
   ```
   This script lets you verify the environment, compute normalization statistics (if needed), launch JAX or PyTorch training, and start the inference server.

3. **docker-compose alternative**
   ```bash
   WANDB_API_KEY=your_wandb_key docker-compose up -d
   docker-compose exec openpi ./setup_and_train.sh
   ```

Mount additional volumes for datasets or other assets as needed. Feel free to customize `docker-compose.yml` for multi-container workflows.