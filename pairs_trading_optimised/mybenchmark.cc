#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <array>
#include <thread>
#include <omp.h>

#define NUM_THREADS 256


using namespace std;

std::vector<float> stock1_prices;
std::vector<float> stock2_prices;


vector<float> readCSV(const string& filename);



void read_prices() {

    string gs_file = "GS.csv";
    string ms_file = "MS.csv";

    stock1_prices = readCSV(gs_file);
    stock2_prices = readCSV(ms_file);

}


vector<float> readCSV(const string& filename){
    std::vector<float> prices;
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

        float adjClose = std::stod(row[5]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<float>& stock1_prices, const std::vector<float>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

//    std::array<float, 671025> spread_sum;
//    std::array<float, 671025> spread_sq_sum;
    vector<float> spread (1256);
    //vector<float> spread_sq_sum (1256);
    vector<int> check(4, 0);
    //vector<thread> threads;

   // spread[0] = stock1_prices[0] - stock2_prices[0];
   // spread_sq_sum[0] = (stock1_prices[0] - stock2_prices[0]) * (stock1_prices[0] - stock2_prices[0]);
#pragma omp parallel for
    for (int i = 0; i < stock1_prices.size(); i++) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }
    vector<float> spread_sum (1256*2);
#pragma omp parallel for
    for(int i = N; i< stock1_prices.size(); i++){
        __m512d sum_vec = _mm512_setzero_pd();
        __m512d sq_sum_vec = _mm512_setzero_pd();

        for (size_t j = i - N; j < i; j += 8) {
            __m512d spread_vec = _mm512_loadu_pd(&spread[j]);
            sum_vec = _mm512_add_pd(sum_vec, spread_vec);
            sq_sum_vec = _mm512_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        }

        double sum_vec_total = _mm512_reduce_add_pd(sum_vec);
        double sq_sum_vec_total = _mm512_reduce_add_pd(sq_sum_vec);

        spread_sum[i * 2] = sum_vec_total;
        spread_sum[(i * 2) + 1] = sq_sum_vec_total;

    }

#pragma omp parallel for
    for (size_t i = N; i < stock1_prices.size(); ++i) {
        int idx = (i*2);
        const float mean = (spread_sum[idx])/ N;
        const float stddev = std::sqrt((spread_sum[idx+1])/ N - mean * mean);
        const float current_spread = stock1_prices[i] - stock2_prices[i];
        const float z_score = (current_spread - mean) / stddev;

        if (z_score > 1.0) {
            //check[0]++;  // Long and Short
        } else if (z_score < -1.0) {
            //check[1]++;  // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            //check[2]++;  // Close positions
        } else {
            //check[3]++;  // No signal
        }

    }
    //cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

}


template<size_t N>
void BM_PairsTradingStrategyOptimized(benchmark::State& state) {
    if (stock1_prices.empty() || stock2_prices.empty()) {
        read_prices();
    }
    //cout<<stock1_prices.size()<<":"<<stock2_prices.size()<<endl;
    for (auto _ : state) {
        pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    }
}

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 8);

BENCHMARK_MAIN();