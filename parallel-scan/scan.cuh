long sequential_scan(double* output, double* input, int length);
float blockscan(int *output, int *input, int length, bool bcao);
float scan(double *output, double *input, int length, bool bcao);

void scanLargeDeviceArray(double *output, double *input, int length, bool bcao);
void scanSmallDeviceArray(double *d_out, double *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(double *output, double *input, int length, bool bcao);
void calc_z(vector<double>& stock1_prices, vector<double>& stock2_prices, vector<double>& spread_sum, vector<double>& spread_sq_sum, vector<int>& check);
__global__ void parallelized_zscore_calculation(
        const double *stock1_prices,
        const double *stock2_prices,
        const double *spread_sum,
        const double *spread_sq_sum,
        int *check,
        size_t N,
        size_t size);