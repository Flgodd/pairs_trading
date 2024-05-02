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

    string gs_file = "RELIANCE.csv";
    string ms_file = "ONGC.csv";

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

        double adjClose = std::stod(row[1]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, 671025> spread_sum;
    std::array<double, 671025> spread_sq_sum;
    //vector<int> check(4, 0);
    //vector<thread> threads;

    const int n = stock1_prices.size();
    const int log2n = static_cast<int>(std::log2(n));

    std::vector<double> spread_sum(n, 0.0);
    std::vector<double> spread_sq_sum(n, 0.0);

#pragma omp parallel
    {
        std::vector<double> local_spread_sum(n, 0.0);
        std::vector<double> local_spread_sq_sum(n, 0.0);

#pragma omp for
        for (int d = 1; d <= log2n; d++) {
            for (int k = (1 << d); k < n; k++) {
                const double current_spread = stock1_prices[k] - stock2_prices[k];
                local_spread_sum[k] = current_spread + local_spread_sum[k - (1 << (d-1))];
                local_spread_sq_sum[k] = (current_spread * current_spread) + local_spread_sq_sum[k - (1 << (d-1))];
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                spread_sum[i] += local_spread_sum[i];
                spread_sq_sum[i] += local_spread_sq_sum[i];
            }
        }
    }

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