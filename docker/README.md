# Autobounds with Docker

Containers contain everything the package needs to run including libraries, system tools, code, and runtime. This is especially helpful in this case since many optimization tools utilized by `autobounds` require compilation.

### Install Docker

Follow the [official guides](https://docs.docker.com/get-docker/) to install Docker.

### Build the Container Image from a Dockerfile

A Dockerfile can be seen as a template for a container instance (e.g., a Python class, rather than an instance of that class). An image is built by executing a Dockerfile, a sequence of required commands. Building creates the instance, or the Docker container that will run on your machine.

Note that we need the Autobounds github repo as a zip file to start the building process. **First, download the autobounds github repo by clicking "Code" -> "Download Zip". Then, move the .zip file to this folder.**

We can build the Autobounds docker container image by running the following code in the terminal, assuming this folder is the working directory.

```
sudo docker build --tag autobounds .
```
Depending on the system, the `sudo` may be omitted. The `--tag` flag specifies the tag/name of the image (in this case, "autobounds"), so that it can be launched with that name in the future

### Run a Container Instance from the Image

The container works as an isolated software environment that, by default, is disconnected from your machine's native environment.

Since our goal is to use a Jupyter notebook that runs inside the container instance, so that we don't need to install the dependencies on our machine, we need to connect the container's isolated environment with our native environment. 

We can do so by exposing, or publishing, a networking port of the container to a port of our local machine. In this case, we connect port 8888 of the container to port 8888 of our own machine. 
```
sudo docker run --interactive --tty --publish 8888:8888 --volume [HOST_DIR]:[CONTAINER_DIR] autobounds
```
The `--interactive` and `--tty` flags facilitate make it easier to workin the container. This command will start a terminal that connects to the Docker container environment. Then, we can directly input code that will be executed inside the container. The `--volume` flag will connect a directory on your host machine, such as `/home/user_name/working_dir` to a newly created directory that will live inside the container, such as `/root/working_dir`. When working inside the container, saving files to [CONTAINER_DIR] will cause those files to appear in [HOST_DIR]. This is important because containers do not inherently have any persistent memory; without this step, work would be lost from run to run. The `autobounds` argument refers to the tag/name of the previously created container.

Lastly, fire up a Juypter notebook inside the container
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

Copy the link returned in the terminal in a browser to start using your Jupyter notebook.
