# Autobounds with Docker

Containers contain everything the package needs to run including libraries, system tools, code, and runtime. This is especially helpful in this case since many optimization tools utilized by Autobounds require compilation.

### Install Docker

Follow the [official guides](https://docs.docker.com/get-docker/) to install Docker.

### Build the Container Image from a Dockerfile

Docker image can be seen as a template for a container instance (e.g., python class v.s. an instance of the python class). An image is built by executing a Dockerfile, a sequence of required commands.

Note that we need the Autobounds github repo as a zip file to start the building process. **First, download the autobounds github repo by clicking "Code" --> "Download Zip". Then, move the .zip file to this folder.**

We can build the Autobounds docker container image by running the following code in the terminal, assuming this folder is the working directory.


```
docker build -t autobounds .
```
```-t``` flag specifices the name/tag of the image.

### Run a Container Instance from the Image

The container works as an isolated software environment that is disconnected your machine's native environment.

Since our goal is to use a jupyter notebook that runs inside the container instance so that we don't need to install the dependencies on our machine, we need to connect the container's isolated environment with our native environment. 

We can do so by exposing a networking port of the container to a port of our local machine. In this case, we connect port 8888 of the container to port 8888 of our own machine. 
```
docker run -it -p 8888:8888 autobounds
```
This command will start a terminal that connects to the docker container environment. Then, we can directly input code that will be executed inside the container.

Lastly, let's fire up a juypter notebook inside the container
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

Copy the link returned in the terminal in a browser to start using jupyter notebook.