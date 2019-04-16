# FROM rocm/rocm-terminal:1.6.4
FROM ubuntu:16.04
FROM hcc_build_image:hcc_build_image
MAINTAINER David Salinas <david.salinas@amd.com>

# Download and install an up to date version of cmake, because compiling
# LLVM has implemented a requirement of cmake v3.4.4 or greater
ARG cmake_prefix=/opt/cmake
ARG cmake_ver_major=3.7
ARG cmake_ver_minor=3.7.2
ARG REPO_RADEON=repo.radeon.com
ARG ROCM_PATH=/opt/rocm

# Install Packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
    curl -sL http://${REPO_RADEON}/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    echo deb [arch=amd64] http://${REPO_RADEON}/rocm/apt/debian/ xenial main | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo \
    comgr \
    vim \
    zlib1g-dev:amd64 \
    zip unzip 

ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/hip
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"

RUN chmod 777 $(find /opt/rocm -type d)

RUN wget https://github.com/github/hub/releases/download/v2.3.0-pre10/hub-linux-386-2.3.0-pre10.tgz
RUN tar -xf hub-linux-386-2.3.0-pre10.tgz
RUN hub-linux-386-2.3.0-pre10/install

RUN useradd -ms /bin/bash jenkins && echo "jenkins:jenkins" | chpasswd && adduser jenkins sudo
RUN echo "jenkins ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/jenkins &&     chmod 0440 /etc/sudoers.d/jenkins
RUN su - jenkins
RUN mkdir -p /home/jenkins && chown -R jenkins:jenkins /home/jenkins
RUN usermod -a -G video jenkins
USER jenkins
