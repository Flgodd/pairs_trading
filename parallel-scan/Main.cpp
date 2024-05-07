#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <iostream>
#include <array>

#include <chrono>
#include <stdlib.h>
#include <stdio.h>

#include <time.h>
#include <cuda_runtime.h> // This usually provides basic atomics
#include <atomic>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "scan.cuh"
#include "utils.h"

using namespace std;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;

vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "Intel.csv";
    string ms_file = "AMD.csv";

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
        double adjClose = std::stod(row[4]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2");
    double spread_sum_f[9866];
    double spread_sq_sum_f[9866];
    vector<int> check(4, 0);

    size_t spread_size = stock1_prices.size();

    fillArrays(stock1_prices, stock2_prices, spread_sum_f, spread_sq_sum_f, spread_size);

    //calc_zz(stock1_prices,stock2_prices,spread_sum_f, spread_sq_sum_f,  check, spread_size);


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

