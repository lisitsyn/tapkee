CXX=g++
CFLAGS=-fPIC -Wall -Weffc++ -Wextra
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: default

default: 
	$(CXX) $(CFLAGS) -O3 -o application src/main.cpp $(SOURCES) -larpack -fopenmp $(LDFLAGS)

noarpack:
	$(CXX) $(CFLAGS) -O3 -o application src/main.cpp $(SOURCES) -DTAPKEE_NO_ARPACK -fopenmp $(LDFLAGS)

debug:
	$(CXX) $(CFLAGS) -g -o application src/main.cpp $(SOURCES) -DTAPKEE_DEBUG -larpack -fopenmp $(LDFLAGS)

doc:
	doxygen doc/Doxyfile

clean:
	rm -f application

.PHONY: all clean

