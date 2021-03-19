FROM ubuntu:latest

RUN mkdir /app
WORKDIR /app

# Python 3.7
RUN apt-get update && \
apt-get install -y python3.7 && \
apt-get install -y python3-pip
RUN pip3 install numpy pandas
RUN pip3 install sklearn colour seaborn matplotlib statsmodels
RUN pip3 install xgboost lightgbm



