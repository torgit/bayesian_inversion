FROM --platform=linux/amd64 peanoframework/gcc:latest

WORKDIR /

RUN git clone --single-branch --branch exahype-aderdg https://gitlab.lrz.de/hpcsoftware/Peano.git
RUN git clone https://github.com/UM-Bridge/umbridge.git /umbridge

RUN . /opt/spack-environment/activate.sh && \
cd /Peano && \
export PYTHONPATH=$PYTHONPATH:/Peano/python && \
libtoolize; aclocal; autoconf; autoheader; cp src/config.h.in .; automake --add-missing && \
CC=gcc CXX=g++ ./configure --enable-particles --enable-exahype --enable-loadbalancing --enable-blockstructured --with-multithreading=omp CXXFLAGS="-Ofast -std=c++20 -fopenmp -Wno-unknown-attributes -Wno-attributes=clang::" LDFLAGS="-fopenmp" && \
make -j4

RUN mkdir /shared /app
RUN mkdir /app/tracers
COPY resources/input.txt /app
COPY resources/LOH.py /Peano/applications/exahype2/exaseis/Cartesian/LOH.py
COPY resources/Elastic.py /Peano/applications/exahype2/exaseis/Cartesian/Elastic.py


COPY resources/activate.sh /opt/spack-environment/activate.sh
RUN . /opt/spack-environment/activate.sh && \
    cd /Peano && \
    export PYTHONPATH=$PYTHONPATH:/Peano/python && \
    cd applications/exahype2/exaseis/Cartesian && python3 LOH.py

COPY resources/loh_server.cpp /app
RUN cd /app && \
    g++ -I/umbridge/lib loh_server.cpp -pthread -o loh_server

RUN { \
      echo '#!/bin/sh' \
      && echo '.' /opt/spack-environment/activate.sh \
      && echo export PYTHONPATH=$PYTHONPATH:/Peano/python \
      && echo 'exec "$@"'; \
    } > /entrypoint.sh \
&& chmod a+x /entrypoint.sh

EXPOSE 4243

RUN chmod -R 777 /Peano
RUN chmod -R 777 /umbridge
RUN chmod -R 777 /app
RUN chmod -R 777 /shared

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/app/loh_server"]