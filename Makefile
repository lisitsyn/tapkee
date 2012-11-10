CXX=g++
CFLAGS=-fPIC -Wall -Weffc++ -Wextra
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: default

default: 
	$(CXX) $(CFLAGS) -O3 -o application src/main.cpp $(SOURCES) -larpack -fopenmp $(LDFLAGS)

debug:
	$(CXX) $(CFLAGS) -g -o application src/main.cpp $(SOURCES) -larpack -fopenmp $(LDFLAGS)


clean:
	rm $(OBJECTS)

.PHONY: all clean

