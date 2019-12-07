FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir -p /project

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    zsh \
    wget \
    git \
    vim

RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN pip install --upgrade pip

ADD requirements.txt /project
RUN pip install -r /project/requirements.txt
WORKDIR /project

CMD ["/bin/zsh"]