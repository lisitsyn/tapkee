#!/bin/sh

mkdir -p build/src/python  # for the couple of .o files

# DOC on the options: https://github.com/wjakob/nanobind/blob/master/src/nb_combined.cpp#L30

# 17 is from nanobind's doc, does it work welll with others?
# TODO tapkee submmodule or external dependency.
g++ src/python/nanobind_extension.cpp -std=c++17 -fvisibility=hidden -DNDEBUG -DNB_COMPACT_ASSERTIONS -I/usr/include/python3.12 -fPIC -I./include -I./src -I/usr/include/eigen3 -I/home/garcia/coding-challenges/ext/nanobind/include -I~/coding-challenges/ext/nanobind/ext/robin_map/include -Os -c -o build/src/python/nanobind_extension.o

g++ ~/coding-challenges/ext/nanobind/src/nb_combined.cpp -std=c++17 -fvisibility=hidden -DNDEBUG -DNB_COMPACT_ASSERTIONS -I/usr/include/python3.12 -fPIC -I/home/garcia/coding-challenges/ext/nanobind/include -I/home/garcia/coding-challenges/ext/nanobind/ext/robin_map/include -O3 -fno-strict-aliasing -ffunction-sections -fdata-sections -c -o build/src/python/libnanobind.o

g++ -shared -Wl,-s -Wl,--gc-sections build/src/python/nanobind_extension.o build/src/python/libnanobind.o -lfmt -o lib/tapkee.cpython-312-x86_64-linux-gnu.so

PYTHONPATH=. python examples/nanobind.py
