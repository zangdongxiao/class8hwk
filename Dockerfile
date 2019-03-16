FROM ubuntu:16.04
MAINTAINER Dongxiao Zang <zangdongxiao@gmail.com>

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib plotly sklearn

ADD mypython.py /
CMD ["python3", "./mypython.py"]
