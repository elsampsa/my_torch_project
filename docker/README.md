Run your preferred combination of tensorflow, torch and cuda in a docker container.

- Install nvidia docker extensions so that you can use GPUs from docker: [docker-driver.bash](docker-driver.bash).
- Build docker image with [make_docker.bash](make_docker.bash).
- Run container with [run_docker.bash](run_docker.bash).  It also mounts you several directories from your host to docker container in order to facilitate notebook usage.
