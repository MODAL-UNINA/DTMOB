FROM ubuntu:22.04

ARG UID=1000
ARG GID=1000

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
        git \
        tzdata \
        wget \
        p7zip-full \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${GID} appuser && \
    useradd -ms /bin/bash --uid ${UID} --gid ${GID} appuser

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
    conda install -n base --override-channels --yes -c conda-forge mamba 'python_abi=*=*cp*' && \
    mamba env create -f environment.yaml -n appenv --yes && \
    conda clean --all && \
    rm -rf ./environment.yaml && \
    echo "Downloading patch to suppress pytorch ResourceWarning" && \
    wget https://github.com/pytorch/pytorch/commit/9b7130b8db62e9e550366419fa33c0f530d80beb.patch -O /tmp/torch-patch.patch && \
    cd /opt/conda/envs/appenv/lib/python3.11/site-packages && \
    git apply /tmp/torch-patch.patch && \
    rm /tmp/torch-patch.patch && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "export MAMBA_ROOT_PREFIX=/opt/conda" >> ~/.bashrc && \
    echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc && \
    echo "conda activate appenv" >> ~/.bashrc

RUN mkdir -p /app/scheduling && \
    chown -R appuser:appuser /app/scheduling

USER appuser

WORKDIR /app/scheduling/

COPY ./scheduling_main /app/scheduling/

RUN 7z x main.7z.001 -o"./" && \
    rm main.7z.* && \
    mkdir highs-install && \
    cd highs-install && \
    curl -L -o HiGHS.v1.7.2.x86_64-linux-gnu-cxx03.tar.gz \
        https://github.com/JuliaBinaryWrappers/HiGHS_jll.jl/releases/download/HiGHS-v1.7.2%2B0/HiGHS.v1.7.2.x86_64-linux-gnu-cxx03.tar.gz && \
    tar -xzf HiGHS.v1.7.2.x86_64-linux-gnu-cxx03.tar.gz && \
    mv bin/highs lib/libhighs.so.1 lib/libhighs.so.1.7.2 /app/scheduling && \
    cd .. && \
    rm -rf highs-install

ENV PATH /opt/conda/envs/appenv/bin:/app/scheduling:$PATH

RUN conda --version
RUN mamba --version
RUN python --version
RUN pip --version