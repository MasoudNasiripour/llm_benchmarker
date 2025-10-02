FROM python:3.11
LABEL authors="Masoud"

WORKDIR /usr/src/
COPY . .

RUN pip3 install -r requirements.txt
