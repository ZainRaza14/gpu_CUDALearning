NVCC        = nvcc
CXX         = g++
NVCCFLAGS   = -O3 -arch=sm_75 -std=c++14 \
              -Xcompiler "-O3 -fopenmp" \
              --generate-line-info
LDFLAGS     = -lcublas -lcurand -lgomp

TARGET      = neural_network

CU_SRCS     = neuralNetwork.cu neuralNetworkTrainer.cu
CPP_SRCS    = dataReader.cpp main.cpp

CU_OBJS     = $(CU_SRCS:.cu=.o)
CPP_OBJS    = $(CPP_SRCS:.cpp=.o)
OBJS        = $(CU_OBJS) $(CPP_OBJS)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) -O3 -std=c++14 -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
