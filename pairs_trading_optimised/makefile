mybenchmark: mybenchmark.cc
				CC mybenchmark.cc -std=c++20 -O3 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark
#-O3 -march=native
#				 g++ mybenchmark.cc -std=c++20 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark

#for openMP		 g++ mybenchmark.cc -std=c++20 -fopenmp -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark
#for immintrin		 g++ mybenchmark.cc -std=c++20 -mavx -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark
		#		g++ mybenchmark.cc -std=c++20 -mavx2 -mfma -fopenmp -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark

clean:
	rm -f mybenchmark *.o




#g++-13 mybenchmark.cc -std=c++20 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -pthread -o mybenchmark -lc++

		#gcc-13 mybenchmark.cc -std=c++20 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark

	#g++ driver.cpp KalmanFilter.cpp -std=c++20 -mavx2 -mfma -lpthread -o kalman

	#g++ mybenchmark.cc KalmanFilter.cpp -std=c++20 -mavx2 -mfma -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -o kalman