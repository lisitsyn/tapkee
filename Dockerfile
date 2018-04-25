FROM ubuntu:xenial

RUN apt-get update && \
    apt-get install libeigen3-dev libarpack2-dev cmake g++ \
    valgrind python-numpy xsltproc libgtest-dev \
    --force-yes -y --no-install-recommends -q0

COPY include /tapkee/include
COPY src /tapkee/src
COPY test /tapkee/test
COPY CMakeLists.txt /tapkee/CMakeLists.txt

RUN cd /tapkee && \
    mkdir -p build && \
    cd build && \
    cmake -DBUILD_TESTS=ON .. && \
    make && \
    ctest -VV

RUN cd /tapkee && \
    python test/generate_swissroll.py 100 && \
    CALLENV='valgrind --leak-check=full --xml=yes --xml-file=/dev/stdout' \
    TAPKEE_ELF='bin/tapkee' \
    INPUT_FILE='input.dat' \
    OUTPUT_FILE='/dev/null' \
    TRANSFORM='xsltproc test/valgrind_tests_transformer.xslt /dev/stdin' \
    test/valgrind_run_all.sh
