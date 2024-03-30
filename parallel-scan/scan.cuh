long sequential_scan(double* output, double* input, int length);
float blockscan(int *output, int *input, int length, bool bcao);
float scan(double *output, double *input, int length, bool bcao);

void scanLargeDeviceArray(double *output, double *input, int length, bool bcao);
void scanSmallDeviceArray(double *d_out, double *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(double *output, double *input, int length, bool bcao);