CXX=clang++
CFLAGS=-fPIC -Wall -Wextra
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: default

default: 
	$(CXX) $(CFLAGS) -fopenmp -O3 -o tapkee_app src/main.cpp $(SOURCES) -isystem /usr/local/include/eigen3 -I./tapkee -larpack $(LDFLAGS)

gpu:
	$(CXX) $(CFLAGS) -DTAPKEE_GPU -fopenmp -O3 -o tapkee_app src/main.cpp $(SOURCES) -isystem /usr/local/include/eigen3 -I./tapkee -larpack -lOpenCL $(LDFLAGS)

noarpack:
	$(CXX) $(CFLAGS) -O3 -o tapkee_app src/main.cpp $(SOURCES) -DTAPKEE_NO_ARPACK -fopenmp $(LDFLAGS)

debug:
	$(CXX) $(CFLAGS) -g -o tapkee_app src/main.cpp $(SOURCES) -I./tapkee -DTAPKEE_DEBUG -larpack -fopenmp $(LDFLAGS)

doc:
	doxygen doc/Doxyfile

clean:
	rm -f tapkee_app

.PHONY: all clean

