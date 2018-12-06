#FROM centos/devtoolset-7-toolchain-centos7
FROM centos:7

# Update centos
RUN yum -y update; yum clean all

# Base
RUN yum -y install git java-1.8.0-openjdk python; yum clean all

# Enable epel-release repositories
RUN yum --enablerepo=extras install -y epel-release

# Install required base build and packaging commands for ROCm
RUN yum -y install \
    bc \
    bridge-utils \
    bison \
    cmake3 \
    devscripts \
    dkms \
    doxygen \
    dpkg \
    dpkg-dev \
    dpkg-perl \
    elfutils-libelf-devel \
    expect \
    file \
    flex \
    gcc-c++ \
    libgcc \
    glibc.i686 \
    libcxx-devel \
    ncurses \
    ncurses-base \
    ncurses-libs \
    numactl-devel \
    numactl-libs \
    libnuma-dev \
    libssh \
    libunwind-devel \
    libunwind \
    llvm \
    llvm-libs \
    make \
    openssl \
    openssl-libs \
    openssl-devel \
    openssh \
    openssh-clients \
    pciutils \
    pciutils-devel \
    pciutils-libs \
    pkgconfig \
    pth \
    qemu-kvm \
    re2c \
    rpm \
    rpm-build \
    subversion \
    sudo \
    time \
    vim \
    wget

# Enable the epel repository for fakeroot
RUN yum --enablerepo=extras install -y fakeroot
RUN yum clean all

# Build and install the currently blessed version of cmake
RUN wget http://compute-artifactory.amd.com/artifactory/compat-source/cmake-3.5.2.tar.gz
RUN tar -xvf cmake-3.5.2.tar.gz
RUN cd cmake-3.5.2 && ./configure && make && make install && cd .. && rm -rf cmake-3.5.2 && rm -rf cmake-3.5.2.tar.gz

# On CentOS, install package centos-release-scl available in CentOS repository:
RUN yum install -y centos-release-scl

# Install the devtoolset-7 collection:
RUN yum install -y devtoolset-7
RUN yum install -y devtoolset-7-libatomic-devel devtoolset-7-elfutils-libelf-devel

# Add the artifactory repo containing the pre-compiled ROCm rpms
RUN yum-config-manager --nogpgcheck --add-repo http://repo.radeon.com/rocm/yum/rpm
RUN yum --enablerepo=\* clean metadata

# Install the ROCm rpms
RUN yum install --nogpgcheck -y hsakmt-roct hsakmt-roct-dev
RUN yum install --nogpgcheck -y hsa-ext-rocr-dev hsa-rocr-dev rocminfo
RUN yum install --nogpgcheck -y atmi hcc hip_base hip_doc hip_hcc hip_samples hsa-amd-aqlprofile 
RUN yum install --nogpgcheck -y rocm-opencl rocm-opencl-devel rocm-smi

# Start using software collections:
RUN scl enable devtoolset-7 bash
RUN source scl_source enable devtoolset-7

