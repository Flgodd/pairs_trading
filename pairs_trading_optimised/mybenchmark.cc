#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>'
#include <iostream>
#include <chrono>
#include <array>
#include <thread>
#include <omp.h>
#include <mutex>


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

        double adjClose = std::stod(row[4]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, 9986> spread;
    //vector<int> check(4, 0);


#pragma omp parallel for
    for (size_t i = 0; i < stock1_prices.size(); ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }



#pragma omp parallel for
    for (size_t i = N; i < stock1_prices.size(); ++i) {
        int start = i-N;

        double sum = spread[start]+spread[start+1]+spread[start+2]+spread[start+3]
                     + spread[start+4]+spread[start+5]+spread[start+6]+spread[start+7];

        double sq_sum = (spread[start]*spread[start]) + (spread[start+1]*spread[start+1])
                        + (spread[start+2]*spread[start+2]) + (spread[start+3]*spread[start+3])
                        + (spread[start+4]*spread[start+4]) + (spread[start+5]*spread[start+5])
                        + (spread[start+6]*spread[start+6]) + (spread[start+7]*spread[start+7]);

        double mean = sum / N;
        double stddev = std::sqrt(sq_sum / N - mean * mean);
        double current_spread = spread[i];
        double z_score = (current_spread - mean) / stddev;

        //check_mutex.lock();
        //std::lock_guard<std::mutex> lock(check_mutex);
        if (z_score > 1.0) {
            //check[0]++;
        } else if (z_score < -1.0) {
            //check[1]++;
        } else if (std::abs(z_score) < 0.8) {
            //check[2]++;
        } else {
            //check[3]++;
        }
        //check_mutex.unlock();
    }


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