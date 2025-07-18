# Autobounds

Autobounds is an advanced software tool designed to automate the calculation of partial identification bounds in causal inference problems.

Developed by researchers at the University of Pennsylvania, Johns Hopkins University, and Princeton University, Autobounds leverages polynomial programming and dual relaxation techniques to compute the sharpest possible bounds on causal effects, even when data is incomplete or mismeasured. You can learn more in their research paper.

This tool is particularly valuable in fields like economics, political science, and epidemiology, where researchers often face challenges such as confounding, selection bias, measurement error, and noncompliance. By automating the process, Autobounds allows for more precise and reliable estimation of causal relationships, facilitating better-informed decision-making.

To get started, download the software via Docker and run the following command to launch Autobounds:
docker run -p 8888:8888 -it gjardimduarte/autolab:v5
This will allow you to easily integrate Autobounds into your causal inference workflows.

To install it in your machine, clone this repo and use python -m pip install .

Development is being currently conduced by Guilherme Duarte, Dean Knox, and Kai Cooper. 
