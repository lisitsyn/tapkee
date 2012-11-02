CXX=g++
CFLAGS=-Wall -O3 -fPIC -Wall -Weffc++ -Wextra
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: hd

hd: 
	$(CXX) $(CFLAGS) -o application src/main.cpp -I/usr/include/atlas -I/usr/include/superlu -lshogun -lsuperlu $(LDFLAGS)

clean:
	rm $(OBJECTS)

.PHONY: all clean

