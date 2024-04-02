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


#define NUM_THREADS 8


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

void parallelUpSweep(vector<double>& x) {
    int n = x.size();
    int numThreads = NUM_THREADS;
    int maxDepth = std::log2(n);
    std::vector<std::thread> threads;

    for (int d = 0; d < maxDepth; ++d) {
        threads.clear();
        int powerOfTwoDPlus1 = std::pow(2, d + 1);

        for (int i = 0; i < numThreads; ++i) {
            int start = i * n / numThreads;
            int end = std::min(n, (i + 1) * n / numThreads);

            start = (start / powerOfTwoDPlus1) * powerOfTwoDPlus1;
            end = ((end + powerOfTwoDPlus1 - 1) / powerOfTwoDPlus1) * powerOfTwoDPlus1;

            threads.emplace_back([=, &x]() {
                for (int k = start; k < end; k += powerOfTwoDPlus1) {
                    double idx1 = k + std::pow(2, d) - 1;
                    double idx2 = k + powerOfTwoDPlus1 - 1;
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
    }
}

void parallelDownSweep(vector<double>& x) {
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
                    double idx1 = k + std::pow(2, d) - 1;
                    double idx2 = k + powerOfTwoDPlus1 - 1;
                    if (idx2 < n) {
                        double tmp = x[idx1];
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
    }
}

void recurive_blelloch(vector<double>& x, int depth){
    //if(depth == 0)return;
    int rem = (x.size()%(NUM_THREADS*2));
    int div = x.size()/(NUM_THREADS*2);
    if(rem != 0){
        rem = (NUM_THREADS*2) - rem;
        for(int i = 0; i<rem; i++){
            x.push_back(0);
        }
        div++;
    }

    int n = NUM_THREADS*2;
    //change to one d vector
    vector<vector<double>> toHoldValues(div);
    vector<double> newX(div);
    for(int i = 0; i<div; i++){
        std::vector<double> temp(x.begin() + (n*i), x.begin() + (n*i + n));

        parallelUpSweep(temp);

        parallelDownSweep(temp);
        toHoldValues[i] = (temp);
        newX[i]=(temp.back() + x[n*i + n-1]);

    }
    double bigg = newX.back();

    if(depth-1==0){
        x = toHoldValues[0];
        return;
    }
    recurive_blelloch(newX, depth-1);

    x.clear();
    //parallelise and using simd
    newX.push_back(newX.back() + bigg);
    for(int i = 0; i<toHoldValues.size(); i++){
        for(int j = 0; j<toHoldValues[i].size(); j++){
            toHoldValues[i][j] += newX[i];
        }
        x.insert(x.end(), toHoldValues[i].begin(), toHoldValues[i].end());
    }
}



template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    vector<double> spread_sum(1256);
    vector<double> spread_sq_sum(1256);
    vector<int> check(4, 0);


    for(int i = 0; i<stock1_prices.size(); i++){
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        spread_sum[i] = current_spread;
        spread_sq_sum[i] = current_spread*current_spread;
    }

    int depth = std::log(spread_sum.size())/log(NUM_THREADS*2);
    float  check_depth = std::log(spread_sum.size())/log(NUM_THREADS*2);
    int rem = (spread_sum.size()%(NUM_THREADS*2));

    if(rem != 0 || check_depth > depth)depth++;


    recurive_blelloch(spread_sum, depth);
    recurive_blelloch(spread_sq_sum, depth);

    for (size_t i = N; i < stock1_prices.size(); ++i) {

        const double mean = (spread_sum[i] - spread_sum[i-N])/ N;
        const double stddev = std::sqrt((spread_sq_sum[i] - spread_sq_sum[i-N])/ N - mean * mean);
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

    cout<<std::thread::hardware_concurrency()<<endl;
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





