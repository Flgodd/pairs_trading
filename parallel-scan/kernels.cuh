__global__ void prescan_arbitrary(double *g_odata, double *g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(double *g_odata, double *g_idata, int n, int powerOfTwo);

__global__ void prescan_large(double *g_odata, double *g_idata, int n, double* sums);
__global__ void prescan_large_unoptimized(double *output, double *input, int n, double *sums);

__global__ void add(double *output, int length, double *n1);
__global__ void add(double *output, int length, double *n1, double *n2);
