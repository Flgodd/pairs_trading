#include <vector>
long sequential_scan(double* output, double* input, int length);
float blockscan(int *output, int *input, int length, bool bcao);
float scan(double *output, double *input, int length, bool bcao,cudaStream_t stream);

void scanLargeDeviceArray(double *output, double *input, int length, bool bcao,cudaStream_t stream);
void scanSmallDeviceArray(double *d_out, double *d_in, int length, bool bcao,cudaStream_t stream);
void scanLargeEvenDeviceArray(double *output, double *input, int length, bool bcao,cudaStream_t stream);
void calc_z(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices,
            const std::vector<double>& spread_sum, const std::vector<double>& spread_sq_sum, std::vector<int>& check);
__global__ void parallelized_zscore_calculation(
        const double *stock1_prices,
        const double *stock2_prices,
        const double *spread_sum,
        const double *spread_sq_sum,
        int *check,
        size_t N,
        size_t size);
__global__ void parallelized_zscore_calculation1(
        const double *stock1_prices,
        const double *stock2_prices,
        const double *spread_sum,
        const double *spread_sq_sum,
        int *check,
        int N,
        size_t size);
void calc_zz(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices,
             double spread_sum[], double spread_sq_sum[],
             std::vector<int>& check, size_t spread_size);
__global__ void para_fill(const double *stock1_prices,
                          const double *stock2_prices,
                          double *spread_sum,
                          double *spread_sq_sum,
                          size_t size);
void fillArrays(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices,
                double spread_sum[], double spread_sq_sum[], size_t spread_size);