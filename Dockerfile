FROM python:3.8.17-slim

ENV TZ='Asia/Singapore' \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

ARG APPDIR=/app
WORKDIR $APPDIR

ARG PIP_TRUSTED_HOST="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

COPY ./requirements.txt ./
RUN pip install --verbose -r requirements.txt --no-cache-dir

# copy application source
COPY . $APPDIR/

EXPOSE 8000

# Run the application:
# ENTRYPOINT [ "/bin/sh" ]
