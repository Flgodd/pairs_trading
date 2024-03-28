#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>'
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


#define NUM_THREADS 4


using namespace std;

namespace simd = std::experimental;

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

void parallelUpSweep(vector<int>& x) {
    int n = x.size();
    int numThreads = 4;
    int maxDepth = std::log2(n);
    std::vector<std::thread> threads;

    for (int d = 0; d < maxDepth; ++d) {
        threads.clear();
        int powerOfTwoDPlus1 = std::pow(2, d + 1);

        for (int i = 0; i < numThreads; ++i) {
            int start = i * n / numThreads;
            int end = std::min(n, (i + 1) * n / numThreads);

            // Adjust start and end to align with powerOfTwoDPlus1 boundaries
            start = (start / powerOfTwoDPlus1) * powerOfTwoDPlus1;
            end = ((end + powerOfTwoDPlus1 - 1) / powerOfTwoDPlus1) * powerOfTwoDPlus1;

            threads.emplace_back([=, &x]() {
                for (int k = start; k < end; k += powerOfTwoDPlus1) {
                    int idx1 = k + std::pow(2, d) - 1;
                    int idx2 = k + powerOfTwoDPlus1 - 1;
                    //if(d==1&&start==0)cout<<idx1<<":"<<idx2<<":"<<x[idx1]+x[idx2]<<endl;
                    if (idx2 < n) {
                        x[idx2] = x[idx1] + x[idx2];
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        numThreads /= 2;
        //if(d==0)cout<<x[0]<<":"<<x[1]<<":"<<x[2]<<":"<<x[3]<<":"<<x[4]<<":"<<x[5]<<":"<<x[6]<<":"<<x[7]<<endl;

        //cout<<d<<"arr"<<x[0]<<":"<<x[1]<<":"<<x[2]<<":"<<x[3]<<":"<<x[4]<<":"<<x[5]<<":"<<x[6]<<":"<<x[7]<<endl;
    }
}

void parallelDownSweep(vector<int>& x) {
    int n = x.size();
    x[n - 1] = 0; // Initialize the last element to 0
    int numThreads = 1;
    int maxDepth = std::log2(n);
    std::vector<std::thread> threads;

    for (int d = maxDepth - 1; d >= 0; --d) {
        threads.clear();
        int powerOfTwoDPlus1 = std::pow(2, d + 1);

        for (int i = 0; i < numThreads; ++i) {
            int start = i * n / numThreads;
            int end = std::min(n, (i + 1) * n / numThreads);

            // Adjust start and end to align with powerOfTwoDPlus1 boundaries
            start = (start / powerOfTwoDPlus1) * powerOfTwoDPlus1;
            end = ((end + powerOfTwoDPlus1 - 1) / powerOfTwoDPlus1) * powerOfTwoDPlus1;

            threads.emplace_back([=, &x]() {
                for (int k = start; k < end; k += powerOfTwoDPlus1) {
                    int idx1 = k + std::pow(2, d) - 1;
                    int idx2 = k + powerOfTwoDPlus1 - 1;
                    if (idx2 < n) {
                        int tmp = x[idx1];
                        x[idx1] = x[idx2];
                        x[idx2] += tmp;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        numThreads *= 2;
        cout<<d<<"arr"<<x[0]<<":"<<x[1]<<":"<<x[2]<<":"<<x[3]<<":"<<x[4]<<":"<<x[5]<<":"<<x[6]<<":"<<x[7]<<endl;

    }
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, 1256> spread_sum;
    std::array<double, 1256> spread_sq_sum;
    vector<int> check(4, 0);
    std::vector<int> x = {1, 2, 3, 4, 5, 6, 7, 8};
    //int n = stock1_prices.size();

    // int num_threads = std::thread::hardware_concurrency(); // Use available hardware threads

    for(int i = 0; i<stock1_prices.size(); i++){
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        spread_sum[i] = current_spread;
        spread_sq_sum[i] = current_spread*current_spread;
    }


    parallelUpSweep(x);
    //cout<<spread_sum[313]<<":"<<stock1_prices[313]-stock2_prices[313]<<endl;
    //cout<<x[0]<<":"<<x[1]<<":"<<x[2]<<":"<<x[3]<<":"<<x[4]<<":"<<x[5]<<":"<<x[6]<<":"<<x[7]<<endl;
    parallelDownSweep(x);



    const double mean = (spread_sum[N-1])/ N;
    const double stddev = std::sqrt((spread_sq_sum[N-1])/ N - mean * mean);
    const double current_spread = stock1_prices[N] - stock2_prices[N];
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


    for (size_t i = N+1; i < stock1_prices.size(); ++i) {

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
