mybenchmark: mm.cpp
		nvcc -o mybenchmark mm.cpp scan.cu kernels.cu utils.cpp -I. -lcudart -std=c++11 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread

clean:
	rm -f mybenchmark *.o