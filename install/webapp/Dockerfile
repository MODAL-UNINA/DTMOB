FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        default-libmysqlclient-dev \
        git \
        tzdata \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./environment.yaml ./

RUN arch=$(uname -m) && \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux" && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL=${MINICONDA_URL}"-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL=${MINICONDA_URL}"-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget ${MINICONDA_URL} -O /tmp/miniconda.sh -q && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda update --all --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda install -n base --override-channels --yes -c conda-forge mamba 'python_abi=*=*cp*' && \
    echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc && \
    mamba env create -f environment.yaml -n appenv --yes && \
    conda clean --all && \
    rm -rf ./environment.yaml && \
    echo "conda activate appenv" >> ~/.bashrc

ENV PATH /opt/conda/envs/appenv/bin:$PATH

RUN conda --version
RUN mamba --version
RUN python --version
RUN pip --version
