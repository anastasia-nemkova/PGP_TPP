NVCC = nvcc
NVCCFLAGS = -O2 -std=c++11 -Xcompiler -Wno-error -w -lm
BIN = kp1
SRC_DIR = src

SRCS = main.cu

all: $(BIN)

$(BIN): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(BIN) -lm

clean:
	rm -f $(BIN) output/*.data

run_default: $(BIN)
	./$(BIN) --default

run_cpu: $(BIN)
	./$(BIN) --cpu

run_gpu: $(BIN)
	./$(BIN) --gpu

.PHONY: all clean run_default run_cpu run_gpu