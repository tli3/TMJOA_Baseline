FROM python:3.7

RUN pip3 install -r requirements.txt

RUN apt-get update && \
	apt-get install -y git

RUN mkdir /app
WORKDIR /app

RUN git clone -b v2.0 https://github.com/celiale/OAI.git
