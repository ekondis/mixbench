FROM ubuntu:rolling

RUN apt-get update \
   && apt-get install -y \
      g++ \
      cmake

ADD . /root/mixbench-cpu

WORKDIR /root/mixbench-cpu

RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-march=native -funroll-loops" -B build-docker ./
RUN cmake --build build-docker

CMD /root/mixbench-cpu/build-docker/mixbench-cpu
