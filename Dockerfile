FROM ubuntu:bionic

ENV ROOT "/app"
ENV PORT 80

RUN apt-get update && apt-get install leiningen -y

RUN mkdir ${ROOT}
WORKDIR ${ROOT}

COPY project.clj .
COPY Procfile .
COPY src src/
COPY resources resources/

EXPOSE 80

CMD ["/usr/bin/lein", "run"]
