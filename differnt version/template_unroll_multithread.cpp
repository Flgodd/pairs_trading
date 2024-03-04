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
#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
#include <arm_neon.h>
#include <array>
#include <thread>


using namespace std;

namespace simd = std::experimental;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);
void read_prices();



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

constexpr size_t unroll_iter = 4;
template<size_t N, size_t Step = 0>
struct UnrollLoop {
    static void process(const std::array<double, N>& spread, double& final_sum, double& final_sq_sum) {
        if constexpr (Step < N) {
            float64x2_t spread_vec = vld1q_f64(&spread[Step]);
            float64x2_t sum_vec = vdupq_n_f64(0.0);
            float64x2_t sq_sum_vec = vdupq_n_f64(0.0);

            sum_vec = vaddq_f64(sum_vec, spread_vec);
            sq_sum_vec = vaddq_f64(sq_sum_vec, vmulq_f64(spread_vec, spread_vec));

            double sum[2], sq_sum[2];
            vst1q_f64(sum, sum_vec);
            vst1q_f64(sq_sum, sq_sum_vec);

            final_sum += sum[0] + sum[1];
            final_sq_sum += sq_sum[0] + sq_sum[1];

            // Recursively call the next step of the unroll
            UnrollLoop<N, Step + 2>::process(spread, final_sum, final_sq_sum);
        }
    }
};

template<size_t N>
void processChunkSIMD(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices, size_t start, size_t end, double& final_sum, double& final_sq_sum) {
    std::array<double, N> spread; // Ensure N is suitable for the chunk size
    for (size_t i = start, j = 0; i < end && j < N; ++i, ++j) {
        spread[j] = stock1_prices[i] - stock2_prices[i];
    }

    UnrollLoop<N>::process(spread, final_sum, final_sq_sum);
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    const size_t numThreads = std::thread::hardware_concurrency();
    const size_t totalSize = stock1_prices.size();
    const size_t chunkSize = (totalSize + numThreads - 1) / numThreads;

    std::vector<std::thread> threads;
    std::vector<double> sums(numThreads, 0.0);
    std::vector<double> sq_sums(numThreads, 0.0);

    for (size_t i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = std::min(start + chunkSize, totalSize);
        threads.emplace_back(processChunkSIMD<N>, std::ref(stock1_prices), std::ref(stock2_prices), start, end, std::ref(sums[i]), std::ref(sq_sums[i]));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    double final_sum = std::accumulate(sums.begin(), sums.end(), 0.0);
    double final_sq_sum = std::accumulate(sq_sums.begin(), sq_sums.end(), 0.0);

    // Use final_sum and final_sq_sum as needed for further processing, like calculating mean and stddev
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
