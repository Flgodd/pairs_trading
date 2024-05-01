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

#define NUM_THREADS 256


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


void calculate_spreads(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices, std::vector<double>& spread_sum, std::vector<double>& spread_sq_sum) {
    int n = stock1_prices.size();
    int num_threads;
#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    std::vector<double> local_starts(num_threads), local_sq_starts(num_threads);

    // Calculate the starting sums for each block
    local_starts[0] = spread_sum[0];
    local_sq_starts[0] = spread_sq_sum[0];
    for (int i = 1; i < num_threads; i++) {
        int start_index = i * (n / num_threads);
        local_starts[i] = local_starts[i - 1] + stock1_prices[start_index] - stock2_prices[start_index] + spread_sum[start_index - 1];
        local_sq_starts[i] = local_sq_starts[i - 1] + (stock1_prices[start_index] - stock2_prices[start_index]) * (stock1_prices[start_index] - stock2_prices[start_index]) + spread_sq_sum[start_index - 1];
    }

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int block_size = n / num_threads;
        int start = thread_id * block_size;
        int end = (thread_id == num_threads - 1) ? n : start + block_size;

        // If not the first block, initialize with the calculated start values
        if (start != 0) {
            spread_sum[start] = local_starts[thread_id];
            spread_sq_sum[start] = local_sq_starts[thread_id];
        }

        for (int i = start + 1; i < end; i++) {
            spread_sum[i] = (stock1_prices[i] - stock2_prices[i]) + spread_sum[i - 1];
            spread_sq_sum[i] = (stock1_prices[i] - stock2_prices[i]) * (stock1_prices[i] - stock2_prices[i]) + spread_sq_sum[i - 1];
        }
    }
}

template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    //std::array<double, 9866> spread_sum;
    //std::array<double, 9866> spread_sq_sum;
    vector<double> spread_sum(1256);
    vector<double> spread_sq_sum(1256);
    vector<int> check(4, 0);
    //vector<thread> threads;

    spread_sum[0] = stock1_prices[0] - stock2_prices[0];
    spread_sq_sum[0] = (stock1_prices[0] - stock2_prices[0]) * (stock1_prices[0] - stock2_prices[0]);

    calculate_spreads(stock1_prices, stock2_prices, spread_sum, spread_sq_sum);

//#pragma omp parallel for
    for (size_t i = N; i < stock1_prices.size(); ++i) {

        const double mean = (spread_sum[i-1] - spread_sum[i-N-1])/ N;
        const double stddev = std::sqrt((spread_sq_sum[i-1] - spread_sq_sum[i-N-1])/ N - mean * mean);
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        const double z_score = (current_spread - mean) / stddev;

        if (z_score > 1.0) {
            check[0]++;  // Long and Short
        } else if (z_score < -1.0) {
            check[1]++;  // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            check[2]++;  // Close positions
        } else {
            check[3]++;  // No signal
        }

    }
    cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

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