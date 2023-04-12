FROM tensorflow/tensorflow AS deeper_build


# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get install ca-certificates && \
    apt-get install -y \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
    libsm6 \
    libspatialindex-dev \
    libxext6 \
    libxrender-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    wget 

# Manually install PIP
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN  python3.10 -m ensurepip 

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1


WORKDIR /root/Deeper

COPY . .

RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install --upgrade requests  \
    && python3.10 -m pip install poetry &&  python3.10 -m poetry install \
    && python3.10 -m pip install .

CMD poetry shell
