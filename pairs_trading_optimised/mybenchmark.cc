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


namespace simd {
    using rf_256 = __m256;  // Using 256-bit wide registers
    using ri_256 = __m256i;  // Using 256-bit wide integer registers

    // Fused multiply-add function using AVX2
    inline rf_256 fused_multiply_add(rf_256 a, rf_256 b, rf_256 c) {
        return _mm256_fmadd_ps(a, b, c);
    }

    template<int KernelSize, int KernelCenter>
    void avx_convolve(const double* stock1_prices, const double* stock2_prices, double (*spread)[2], size_t len) {
        static_assert(KernelSize > 1 && KernelSize <= 16);
        static_assert(KernelCenter >= 0 && KernelCenter < KernelSize);
        constexpr int WindowCenter = KernelSize - KernelCenter - 1;

        // Adjusted for 256-bit wide registers
        rf_256 prev1, prev2;
        rf_256 curr1, curr2;
        rf_256 next1, next2;

        prev1 = _mm256_set1_ps(0.0f);
        prev2 = _mm256_set1_ps(0.0f);
        curr1 = _mm256_loadu_ps(stock1_prices);
        curr2 = _mm256_loadu_ps(stock2_prices);
        next1 = _mm256_loadu_ps(stock1_prices + 8);  // Adjusted stride for 256-bit
        next2 = _mm256_loadu_ps(stock2_prices + 8);  // Adjusted stride for 256-bit

        for (auto pEnd = stock1_prices + len - 8; stock1_prices < pEnd; stock1_prices += 8, stock2_prices += 8) {  // Adjusted increment for 256-bit
            double sum[2] = {0.0f, 0.0f};

            for (int k = 0; k < KernelSize; ++k)
            {
                rf_256 diff = _mm256_sub_ps(_mm256_set1_ps(stock1_prices[k]), _mm256_set1_ps(stock2_prices[k]));
                rf_256 diff_prev = _mm256_sub_ps(_mm256_set1_ps(stock1_prices[k-KernelSize]), _mm256_set1_ps(stock2_prices[k-KernelSize]));

                sum[0] += _mm256_reduce_add_ps(fused_multiply_add(diff, _mm256_set1_ps(1.0), _mm256_set1_ps(sum[0])));
                sum[0] += _mm256_reduce_add_ps(fused_multiply_add(diff_prev, _mm256_set1_ps(-1.0), _mm256_set1_ps(sum[0])));

                sum[1] += _mm256_reduce_add_ps(fused_multiply_add(_mm256_mul_ps(diff, diff), _mm256_set1_ps(1.0), _mm256_set1_ps(sum[1])));
                sum[1] += _mm256_reduce_add_ps(fused_multiply_add(_mm256_mul_ps(diff_prev, diff_prev), _mm256_set1_ps(-1.0), _mm256_set1_ps(sum[1])));
            }

            spread[stock1_prices - stock1_prices][0] = sum[0];
            spread[stock1_prices - stock1_prices][1] = sum[1];

            prev1 = curr1;
            prev2 = curr2;
            curr1 = next1;
            curr2 = next2;
            next1 = _mm256_loadu_ps(stock1_prices + 16);  // Adjusted stride for 256-bit
            next2 = _mm256_loadu_ps(stock2_prices + 16);  // Adjusted stride for 256-bit
        }
    }
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    //std::array<std::array<double, 2>, 1256> spread;
    double spread[1256][2];  // Changing to a simple C-style array for compatibility

    // Initialize the first N elements of the spread array
    for (size_t i = 0; i < N; ++i) {
        double current_spread = stock1_prices[i] - stock2_prices[i];
        spread[i][0] = current_spread + (i > 0 ? spread[i-1][0] : 0.0);
        spread[i][1] = current_spread * current_spread + (i > 0 ? spread[i-1][1] : 0.0);
    }

    // Call the avx_convolve function to calculate the spread for the remaining elements
    simd::avx_convolve<N, N/2>(&stock1_prices[N], &stock2_prices[N], spread + N, 1256 - N);

    for (size_t i = N; i < stock1_prices.size(); ++i) {
        double mean = spread[i-1][0] / N;
        double stddev = std::sqrt(spread[i-1][1] / N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;

        if (z_score > 1.0) {
            // Long and Short
        } else if (z_score < -1.0) {
            // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            // Close positions
        } else {
            // No signal
        }
    }
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