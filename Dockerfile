FROM tensorflow/tensorflow AS deeper_build


WORKDIR /root/Deeper


RUN ls -all

RUN pip install poetry && \
    poetry install && \
    poetry shell && \
    python -m unittest #&& \
    #poetry build

CMD poetry shell
