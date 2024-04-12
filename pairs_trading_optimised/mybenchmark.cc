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

#define NUM_THREADS 16


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

/*void block_prefix_sum(const int start, const int end, const vector<double>& stock1_prices, const vector<double>& stock2_prices, array<double, 1256>& spread_sum, array<double, 1256>& spread_sq_sum) {

    for (int i = start+1; i <= end; i++) {
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        spread_sum[i] = current_spread + spread_sum[i - 1];
        spread_sq_sum[i] = (current_spread * current_spread) + spread_sq_sum[i - 1];
    }
    return;
}*/


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, 1256> spread_sum;
    std::array<double, 1256> spread_sq_sum;
    //vector<int> check(4, 0);
    vector<thread> threads;

    spread_sum[0] = stock1_prices[0] - stock2_prices[0];
    spread_sq_sum[0] = (stock1_prices[0] - stock2_prices[0]) * (stock1_prices[0] - stock2_prices[0]);

    auto worker = [&](size_t start, size_t end) {
        for (int i = start + 1; i <= end; i++) {
            const double current_spread = stock1_prices[i] - stock2_prices[i];
            spread_sum[i] = current_spread + spread_sum[i - 1];
            spread_sq_sum[i] = (current_spread * current_spread) + spread_sq_sum[i - 1];
        }
    };

    int blockSize = stock1_prices.size() / NUM_THREADS;
    int remainingSize = stock1_prices.size() % NUM_THREADS;

    int start = 0;
    int end = blockSize-1;
#pragma omp simd
    for (int i = 0; i < NUM_THREADS; i++) {
        if(i < remainingSize)end++;
        const double current_spread = stock1_prices[start] - stock2_prices[start];
        spread_sum[start] = current_spread;
        spread_sq_sum[start] = current_spread * current_spread;

        threads.push_back(thread(worker, start, end));

        start = end +1;
        end = start+blockSize-1;
    }

    for (auto& th : threads) {
        th.join();
    }

    start = (remainingSize == 0) ? blockSize : blockSize+1;
    end = start+blockSize-1;;
    for (int i = 1; i < NUM_THREADS; i++) {

        if(i < remainingSize)end++;
        for (int j = start; j <= end; j++) {
            spread_sum[j] += spread_sum[start - 1];
            spread_sq_sum[j] += spread_sq_sum[start - 1];
        }
        start = end + 1;
        end = start+blockSize-1;
    }

    const double mean = (spread_sum[N-1])/ N;
    const double stddev = std::sqrt((spread_sq_sum[N-1])/ N - mean * mean);
    const double current_spread = stock1_prices[N] - stock2_prices[N];
    const double z_score = (current_spread - mean) / stddev;


    if (z_score > 1.0) {
        //check[0]++;  // Long and Short
    } else if (z_score < -1.0) {
        //check[1]++;  // Short and Long
    } else if (std::abs(z_score) < 0.8) {
        //check[2]++;  // Close positions
    } else {
        //check[3]++;  // No signal
    }

#pragma omp parallel for
    for (size_t i = N+1; i < stock1_prices.size(); ++i) {

        const double mean = (spread_sum[i-1] - spread_sum[i-N-1])/ N;
        const double stddev = std::sqrt((spread_sq_sum[i-1] - spread_sq_sum[i-N-1])/ N - mean * mean);
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        const double z_score = (current_spread - mean) / stddev;

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
    for (auto _ : state) {
        pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    }
}

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 8);

BENCHMARK_MAIN();
