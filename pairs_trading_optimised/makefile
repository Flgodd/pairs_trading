mybenchmark: mybenchmark.cc
	g++ -std=c++20 -D_GLIBCXX_PARALLELISM_TS -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark mybenchmark.cc

	#g++ mybenchmark.cc -std=c++20 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark

clean:
	rm -f mybenchmark *.o