FROM ubuntu:bionic

ENV ROOT "/app"

RUN apt-get update && apt-get install leiningen -y

RUN mkdir ${ROOT}
WORKDIR ${ROOT}

COPY project.clj .
COPY Procfile .
COPY src src/
COPY resources resources/

CMD ["/usr/bin/lein", "run"]
