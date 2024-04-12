#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>'
#include <iostream>
#include <array>
#include <chrono>
#include <thread>
#include <omp.h>


#define NUM_THREADS omp_get_max_threads()


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

void parallelUpSweep(std::vector<double>& x) {
    const int n = x.size();
    const int maxDepth = std::log2(n);

    for (int d = 0; d < maxDepth; ++d) {
        const int powerOfTwoDPlus1 = 1 << (d + 1);

#pragma omp parallel for
        for (int k = 0; k < n; k += powerOfTwoDPlus1) {
            const int idx1 = k + (1 << d) - 1;
            const int idx2 = k + powerOfTwoDPlus1 - 1;
            if (idx2 < n) {
                x[idx2] += x[idx1];
            }
        }
    }
}

void parallelDownSweep(std::vector<double>& x) {
    const int n = x.size();
    x[n - 1] = 0; // Initialize the last element to 0
    const int maxDepth = std::log2(n);

    for (int d = maxDepth - 1; d >= 0; --d) {
        const int powerOfTwoDPlus1 = 1 << (d + 1);

#pragma omp parallel for
        for (int k = 0; k < n; k += powerOfTwoDPlus1) {
            const int idx1 = k + (1 << d) - 1;
            const int idx2 = k + powerOfTwoDPlus1 - 1;
            if (idx2 < n) {
                const double tmp = x[idx1];
                x[idx1] = x[idx2];
                x[idx2] += tmp;
            }
        }
    }
}

void recursive_blelloch(std::vector<double>& x, int depth) {
    const int n = omp_get_max_threads() * 2;
    const int size = x.size();
    const int paddedSize = ((size + n - 1) / n) * n;
    x.resize(paddedSize, 0);

    const int div = paddedSize / n;
    std::vector<std::vector<double>> toHoldValues(div);
    std::vector<double> newX(div);

#pragma omp parallel for
    for (int i = 0; i < div; ++i) {
        toHoldValues[i].assign(x.begin() + i * n, x.begin() + (i + 1) * n);
        parallelUpSweep(toHoldValues[i]);
        parallelDownSweep(toHoldValues[i]);
        newX[i] = toHoldValues[i].back() + x[(i + 1) * n - 1];
    }

    if (depth == 1) {
        x = std::move(toHoldValues[0]);
        x.resize(size);
        return;
    }

    recursive_blelloch(newX, depth - 1);

#pragma omp parallel for
    for (int i = 0; i < div; ++i) {
        for (int j = 0; j < n; ++j) {
            toHoldValues[i][j] += newX[i];
        }
    }

    x.clear();
    for (const auto& subvec : toHoldValues) {
        x.insert(x.end(), subvec.begin(), subvec.end());
    }
    x.resize(size);
}



template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    vector<double> spread_sum(1256);
    vector<double> spread_sq_sum(1256);
    //vector<int> check(4, 0);

#pragma omp parallel for
    for(int i = 0; i<stock1_prices.size(); i++){
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        spread_sum[i] = current_spread;
        spread_sq_sum[i] = current_spread*current_spread;
    }

    int depth = std::log(spread_sum.size())/log(NUM_THREADS*2);
    float  check_depth = std::log(spread_sum.size())/log(NUM_THREADS*2);
    int rem = (spread_sum.size()%(NUM_THREADS*2));

    if(rem != 0 || check_depth > depth)depth++;


    recursive_blelloch(spread_sum, depth);
    recursive_blelloch(spread_sq_sum, depth);
#pragma omp parallel for
    for (size_t i = N; i < stock1_prices.size(); ++i) {

        const double mean = (spread_sum[i] - spread_sum[i-N])/ N;
        const double stddev = std::sqrt((spread_sq_sum[i] - spread_sq_sum[i-N])/ N - mean * mean);
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

    //std::cout << "Maximum number of threads = " << omp_get_max_threads() << std::endl;

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
