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


#define NUM_THREADS 8


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

void parallelUpSweep(vector<double>& x) {
    int n = x.size();
    int numThreads = NUM_THREADS;
    int maxDepth = std::log2(n);
    std::vector<std::thread> threads;

    for (int d = 0; d < maxDepth; ++d) {
        threads.clear();
        int powerOfTwoDPlus1 = 1 << (d + 1);

        for (int i = 0; i < numThreads; ++i) {
            int start = i * n / numThreads;
            int end = std::min(n, (i + 1) * n / numThreads);

            start = (start / powerOfTwoDPlus1) * powerOfTwoDPlus1;
            end = ((end + powerOfTwoDPlus1 - 1) / powerOfTwoDPlus1) * powerOfTwoDPlus1;

            threads.emplace_back([=, &x]() {
                for (int k = start; k < end; k += powerOfTwoDPlus1) {
                    double idx1 = k + (1 << d) - 1;
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
        int powerOfTwoDPlus1 = 1 << (d + 1);
        int chunkSize = (n + numThreads - 1) / numThreads;
        chunkSize = (chunkSize + powerOfTwoDPlus1 - 1) / powerOfTwoDPlus1 * powerOfTwoDPlus1;

        threads.resize(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            int start = i * chunkSize;
            int end = std::min(start + chunkSize, n);

            threads[i] = std::thread([=, &x]() {
                for (int k = start; k < end; k += powerOfTwoDPlus1) {
                    int idx1 = k + (1 << d) - 1;
                    int idx2 = std::min(k + powerOfTwoDPlus1 - 1, n - 1);
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

void recurive_blelloch(vector<double>& x, int depth) {
    int rem = x.size() % (NUM_THREADS * 2);
    int div = x.size() / (NUM_THREADS * 2);
    if (rem != 0) {
        rem = (NUM_THREADS * 2) - rem;
        x.resize(x.size() + rem, 0);
        div++;
    }

    int n = NUM_THREADS * 2;
    vector<vector<double>> toHoldValues(div);
    vector<double> newX(div);

    vector<thread> threads;
    for (int i = 0; i < div; i++) {
        threads.emplace_back([&, i] {
            toHoldValues[i].resize(n);
            std::copy(x.begin() + (n * i), x.begin() + (n * i + n), toHoldValues[i].begin());

            parallelUpSweep(toHoldValues[i]);
            parallelDownSweep(toHoldValues[i]);

            newX[i] = toHoldValues[i].back() + x[n * i + n - 1];
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    double bigg = newX.back();

    if (depth - 1 == 0) {
        x = std::move(toHoldValues[0]);
        return;
    }

    recurive_blelloch(newX, depth - 1);

    x.clear();
    x.reserve(div * n);

    newX.push_back(newX.back() + bigg);

    threads.clear();
    for (int i = 0; i < div; i++) {
        threads.emplace_back([&, i] {
            double offset = newX[i];
            std::transform(toHoldValues[i].begin(), toHoldValues[i].end(), toHoldValues[i].begin(),
                           [offset](double val) { return val + offset; });
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (const auto& vec : toHoldValues) {
        x.insert(x.end(), vec.begin(), vec.end());
    }
}



template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    size_t size = stock1_prices.size();
    vector<double> spread_sum(size);
    vector<double> spread_sq_sum(size);
    //vector<int> check(4, 0);

    // Parallelize the spread calculation
    size_t num_threads = NUM_THREADS;
    std::vector<std::thread> threads;
    size_t chunk_size = size / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? size : start + chunk_size;

        threads.emplace_back([&, start, end]() {
            for (size_t j = start; j < end; ++j) {
                const double current_spread = stock1_prices[j] - stock2_prices[j];
                spread_sum[j] = current_spread;
                spread_sq_sum[j] = current_spread * current_spread;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int depth = std::log(size) / log(NUM_THREADS * 2);
    float check_depth = std::log(size) / log(NUM_THREADS * 2);
    int rem = (size % (NUM_THREADS * 2));

    if (rem != 0 || check_depth > depth)
        depth++;

    recurive_blelloch(spread_sum, depth);
    recurive_blelloch(spread_sq_sum, depth);

    // Parallelize the z-score calculation and position checks
    // std::vector<std::vector<int>> local_checks(num_threads, std::vector<int>(4, 0));

    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? size : start + chunk_size;

        threads.emplace_back([&, start, end]() {
            for (size_t j = std::max(start, N); j < end; ++j) {
                const double mean = (spread_sum[j] - spread_sum[j - N]) / N;
                const double stddev = std::sqrt((spread_sq_sum[j] - spread_sq_sum[j - N]) / N - mean * mean);
                const double current_spread = stock1_prices[j] - stock2_prices[j];
                const double z_score = (current_spread - mean) / stddev;

                if (z_score > 1.0) {
                    //local_checks[i][0]++; // Long and Short
                } else if (z_score < -1.0) {
                    //local_checks[i][1]++; // Short and Long
                } else if (std::abs(z_score) < 0.8) {
                    //local_checks[i][2]++; // Close positions
                } else {
                    //local_checks[i][3]++; // No signal
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    /*// Combine the local check results
    for (const auto& local_check : local_checks) {
        for (size_t i = 0; i < 4; ++i) {
            check[i] += local_check[i];
        }
    }

    cout << check[0] << ":" << check[1] << ":" << check[2] << ":" << check[3] << endl;*/
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
