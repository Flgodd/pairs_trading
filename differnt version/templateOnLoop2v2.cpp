#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <chrono>
#include <array>


using namespace std;


std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "GS.csv";
    string ms_file = "MS.csv";

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

        double adjClose = std::stod(row[5]);
        prices.push_back(adjClose);
    }


    return prices;
}

template<size_t Index, size_t N>
struct LoopUnroll {
    static void unroll(const std::array<double, N>& spread, __m256d& sum_vec, __m256d& sq_sum_vec) {
        if constexpr (Index < N) {
            __m256d spread_vec = _mm256_loadu_pd(&spread[Index]);
            sum_vec = _mm256_add_pd(sum_vec, spread_vec);
            sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));
            LoopUnroll<Index + 4, N>::unroll(spread, sum_vec, sq_sum_vec); // Increment by 4
        }
    }
};


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 4 == 0, "N should be a multiple of 4 for AVX instructions");
    std::array<double, N> spread;
    size_t spread_index = 0;
    for (size_t i = 0; i < N; ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }
    //vector<int> check(4, 0);
    for (size_t i = N; i < stock1_prices.size(); ++i) {
        __m256d sum_vec = _mm256_setzero_pd();
        __m256d sq_sum_vec = _mm256_setzero_pd();
        LoopUnroll<0, N>::unroll(spread, sum_vec, sq_sum_vec);
        double sum[4], sq_sum[4];
        _mm256_storeu_pd(sum, sum_vec);
        _mm256_storeu_pd(sq_sum, sq_sum_vec);
        double final_sum = sum[0] + sum[1] + sum[2] + sum[3];
        double final_sq_sum = sq_sum[0] + sq_sum[1] + sq_sum[2] + sq_sum[3];
        double mean = final_sum / N;
        double stddev = std::sqrt(final_sq_sum / N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;
        spread[spread_index] = current_spread;
        if (z_score > 1.0) {
            // Long and Short
            //check[0]++;
        }
        else if (z_score < -1.0) {
            // Short and Long
            //check[1]++;
        }
        else if (std::abs(z_score) < 0.8) {
            // Close positions
            //check[2]++;
        }
        else {
            // No signal
            //check[3]++;
        }
        spread_index = (spread_index + 1) % N;
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