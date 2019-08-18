# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
#FROM python:alpine
FROM ubuntu:18.10

# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL maintainer="Parag Pradhan <paragajg@gmail.com>"

#EXPOSE 5000

COPY /src /app
WORKDIR /app
#ADD . /src

# Using pip:
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
#RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
#RUN pip install -U pip

RUN  python3 -m pip install -r requirements.txt
CMD ["python3", "-m", "mlaas_api.py"]

# Using pipenv:
#RUN python3 -m pip install pipenv
#RUN pipenv install --ignore-pipfile
#CMD ["pipenv", "run", "python3", "-m", "app"]

# Using miniconda (make sure to replace 'myenv' w/ your environment name):
#RUN conda env create -f environment.yml
#CMD /bin/bash -c "source activate myenv && python3 -m app"
