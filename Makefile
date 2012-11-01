CXX=clang
CFLAGS=-Wall -O3 -fPIC
SOURCES=$(wildcard src/**/*.cpp)
OBJECTS=$(SOURCES:%.cpp=%.o)

all: hd

hd: 
	$(CXX) -fPIC -fno-exceptions -fno-rtti -o application src/main.cpp -I/usr/include/atlas -I/usr/include/superlu -lshogun -lsuperlu -O3 $(LDFLAGS)

clean:
	rm $(OBJECTS)

.PHONY: all clean

