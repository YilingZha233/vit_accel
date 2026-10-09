CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
PYTHON ?= python3

.PHONY: all test vectors
all: build/bench_matmul

build/bench_matmul: software/hps/bench_matmul.c
	mkdir -p build
	$(CC) $(CFLAGS) -DBUILD_FLAGS='"$(CFLAGS)"' $< -o $@ $(LDLIBS)

test:
	$(PYTHON) -m unittest discover -s tests -v

vectors:
	$(PYTHON) scripts/generate_vectors.py
