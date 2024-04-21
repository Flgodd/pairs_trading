#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <iostream>
#include <array>
//#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
//#include <arm_neon.h>
#include <array>
#include <stdlib.h>
#include <stdio.h>

#include <time.h>
#include <cuda_runtime.h> // This usually provides basic atomics
#include <atomic>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>

#include "scan.cuh"
#include "utils.h"

using namespace std;
const int N = 8;
void test(double in[]) {
    int NN  = 1256;
	bool canBeBlockscanned = NN <= 1024;

	time_t t;
	srand((unsigned)time(&t));
	/*int *in = new int[N];
	for (int i = 0; i < N; i++) {
		in[i] = i+1;//rand() % 10;
	}*/

	printf("%i Elements \n", NN);

		/*// sequential scan on CPU
		double *outHost = new double[NN]();
		long time_host = sequential_scan(outHost, in, NN);
		printResult("host    ", outHost[NN - 1], time_host);*/
		/*// full scan
		double *outGPU = new double[NN]();
		float time_gpu = scan(outGPU, in, NN, false);
		printResult("gpu     ", outGPU[NN - 1], time_gpu);*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
		// full scan with BCAO
		double *outGPU_bcao = new double[NN]();
		float time_gpu_bcao = scan(outGPU_bcao, in, NN, true);
		printResult("gpu bcao", in[NN - 1], time_gpu_bcao);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout<<"in test: "<<elapsedTime<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
		/*if (canBeBlockscanned) {
			// basic level 1 block scan
			int *out_1block = new int[N]();
			float time_1block = blockscan(out_1block, in, N, false);
			printResult("level 1 ", out_1block[N - 1], time_1block);

			// level 1 block scan with BCAO
			int *out_1block_bcao = new int[N]();
			float time_1block_bcao = blockscan(out_1block_bcao, in, N, true);
			printResult("l1 bcao ", out_1block_bcao[N - 1], time_1block_bcao);

			delete[] out_1block;
			delete[] out_1block_bcao;
		}*/

	printf("\n");

    /*for (int i = 0; i < 1256; i++) {
        //if(outHost[i] != outGPU[i])cout<<"outHost:"<<outHost[i]<<" outGPU:"<<outGPU[i]<<endl;
        in[i] = outGPU_bcao[i];
    }*/
	//delete[] outHost;
	//delete[] outGPU;
	delete[] outGPU_bcao;
}

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "GS.csv";
    string ms_file = "MS.csv";

    stock1_prices = readCSV(gs_file);
    stock2_prices = readCSV(ms_file);

}

void calc_z(vector<double>& stock1_prices, vector<double>& stock2_prices, vector<double>& spread_sum, vector<double>& spread_sq_sum, vector<int>& check);



vector<double> readCSV(const string& filename){
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(value);
        }
        //chenge to 4 for intel and amd and 5 for GS and MS
        double adjClose = std::stod(row[5]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");
    //1256 : 9866
    /*cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);*/
    const int NN  = 1256;
    double spread_sum_f[1256];
    double spread_sq_sum_f[1256];
    vector<int> check(4, 0);

    size_t spread_size = stock1_prices.size();

    fillArrays(stock1_prices, stock2_prices, spread_sum_f, spread_sq_sum_f, spread_size);

    cout<<spread_size<<endl;
    double *outGPU_bcao = new double[NN]();
    float time_gpu_bcao = scan(outGPU_bcao, spread_sum_f, NN, true);
    //printResult("gpu bcao", spread_sum_f[NN - 1], time_gpu_bcao);

    float time_gpu_bcao2 = scan(outGPU_bcao, spread_sq_sum_f, NN, true);
   //printResult("gpu bcao", spread_sq_sum_f[NN - 1], time_gpu_bcao2);

    calc_zz(stock1_prices,stock2_prices,spread_sum_f, spread_sq_sum_f,  check, spread_size);


    //cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

}

template<size_t N>
void BM_PairsTradingStrategyOptimized(benchmark::State& state) {
    if (stock1_prices.empty() || stock2_prices.empty()) {
        read_prices();
    }
    for (auto _ : state) {
        pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    }
}

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 8);

BENCHMARK_MAIN();
/*
int main()
{
    read_prices();
    cout<<stock1_prices.size()<<":"<<stock2_prices.size()<<endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout<<"pairs_trading: "<<elapsedTime<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	return 0;
}*/
