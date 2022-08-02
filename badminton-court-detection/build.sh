#!/usr/bin/env sh

set -x

mkdir -p build
cd build

cmake ..
cmake --build .