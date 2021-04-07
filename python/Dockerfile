FROM python:3.7

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN git clone -b v2.0 https://github.com/celiale/OAI.git

WORKDIR /app/python

# # Python 3.7
# RUN apt-get update && \
# apt-get install -y python3.7 && \
# apt-get install -y python3-pip
# RUN pip3 install numpy pandas
# RUN pip3 install sklearn colour seaborn matplotlib statsmodels
# RUN pip3 install xgboost lightgbm



