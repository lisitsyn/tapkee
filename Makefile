CXX=g++
CFLAGS=-Wall -fPIC -Wall -Weffc++ -Wextra
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: default

default: 
	$(CXX) $(CFLAGS) -O3 -o application src/main.cpp $(SOURCES) -I/usr/include/atlas -I/usr/include/superlu -lshogun -lsuperlu -larpack $(LDFLAGS)

debug:
	$(CXX) $(CFLAGS) -g -o application src/main.cpp $(SOURCES) -I/usr/include/atlas -I/usr/include/superlu -lshogun -lsuperlu -larpack $(LDFLAGS)


clean:
	rm $(OBJECTS)

.PHONY: all clean

