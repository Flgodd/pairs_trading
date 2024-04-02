mybenchmark: mybenchmark.cc
		 g++ mybenchmark.cc -std=c++20 -mavx -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark


clean:
	rm -f mybenchmark *.o




#g++-13 mybenchmark.cc -std=c++20 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -pthread -o mybenchmark -lc++

		#gcc-13 mybenchmark.cc -std=c++20 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark