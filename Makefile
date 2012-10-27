CXX=g++
CFLAGS=-Wall -O3 -fPIC
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: hd

hd: 
	g++ -fPIC -o application src/main.cpp $(SOURCES) -I/usr/include/superlu -lsuperlu -O3 -pthread $(LDFLAGS)

clean:
	rm $(OBJECTS)

.PHONY: all clean

