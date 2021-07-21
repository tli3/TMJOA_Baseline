FROM python:3.7

RUN apt-get update && \
	apt-get install -y git

RUN mkdir /app
WORKDIR /app

RUN wget https://github.com/celiale/OAI/archive/refs/tags/3.1.zip
RUN unzip 3.1.zip
RUN mv OAI-3.1 OAI
RUN rm -rf 3.1.zip \
	rm -rf OAI-3.1

WORKDIR /app/OAI/python
RUN pip3 install -r ./requirements.txt
RUN wget https://github.com/celiale/OAI/releases/download/3.1/Models.zip
RUN unzip Models.zip
