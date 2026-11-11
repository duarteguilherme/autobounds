From ubuntu:20.04

MAINTAINER Anonymous
LABEL DESCRIPTION "Autobounds"

# Prevent cmake from asking for user input
ARG DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/lib:/usr/lib:/usr/local/lib


# libjpeg-dev 



RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential \
                                              gfortran \
                                              file \
                                              wget \
                                              unzip \
                                              zlib1g-dev \
                                              bison \
                                              flex \
                                              libgmp-dev \
                                              libreadline-dev \
                                              libncurses5-dev \
                                              glpk-utils \
                                              libblas-dev \
                                              liblapack-dev \
                                              python3-dev \
                                              python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get install -y liblapack-dev zlib1g-dev
RUN set -ex && apt-get update && \
    apt-get install -y cmake


# SCIP
COPY scipoptsuite-8.0.3.tgz /root/
RUN tar xvzf root/scipoptsuite-8.0.3.tgz --directory root/ &&\
     mkdir root/scipoptsuite-8.0.3/build &&\
     cmake root/scipoptsuite-8.0.3 -DAUTOBUILD=on &&\
     make &&\
     make check &&\
     make install


# Install python packages
RUN pip3 install wheel setuptools && \
    pip3 install jupyterlab pyomo notebook &&\
    pip3 install matplotlib numpy scipy pandas &&\
    pip3 install pandas statsmodels


# Autobounds
RUN mkdir /root/autobound
COPY autobounds-main.zip /root/autobound
RUN unzip /root/autobound/autobounds-main.zip -d /root/autobound/
RUN pip3 install /root/autobound/autobounds-main/.


# update indices
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt update -qq
# install two helper packages we need
RUN apt install -y --no-install-recommends software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"



RUN apt update && apt install -y --no-install-recommends r-base-core libpng-dev

# RUN apt install -y libpng-dev
RUN Rscript -e 'install.packages(c("ggplot2", "data.table", "gtools", "foreach", "binom", "geometry", "rgl"))'
RUN Rscript -e 'install.packages(c("igraph","shiny","rcdd"))'
RUN Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/causaloptim/causaloptim_0.8.2.tar.gz")'
RUN Rscript -e 'install.packages("doParallel")'


WORKDIR /root
EXPOSE 8888
ENTRYPOINT ["/bin/bash"]
