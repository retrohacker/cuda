#include <stdio.h>
#define M 3
#define N 3
#define P 3

__global__ void kernel(float*, float*, float*);

int main(int argc, char** argv) {
  /**
   * Init all variables
   */
  int a_size      = sizeof(float) * M * N,
      b_size      = sizeof(float) * N * P,
      result_size = sizeof(float) * M * P;

  float a[]        = {1,2,3,4,5,6,7,8,9},
        b[]        = {9,8,7,6,5,4,3,2,1},
        answer[]   = {30,24,18,84,69,54,138,114,90},
        *result    = (float*) malloc(result_size),
        *d_a,
        *d_b,
        *d_result;

  /**
   * Setup device memory
   */
  cudaMalloc((void**)&d_a,a_size);
  cudaMalloc((void**)&d_b,b_size);
  cudaMalloc((void**)&d_result,result_size);
  cudaMemcpy(d_a,a,a_size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,b_size,cudaMemcpyHostToDevice);

  /**
   * Start GPU
   */
  kernel<<<1,1>>>(d_a,d_b,d_result);

  /**
   * Copy results back to host
   */
  cudaMemcpy(result,d_result,sizeof(float)* M * P,cudaMemcpyDeviceToHost);

  /**
   * Print results
   */
  int i,j;
  for(i=0;i<M;i++) {
    for(j=0;j<P;j++)
      printf("%u ",a[i*M+j]);
    printf("\n");
  }

  /**
   * Cleanup memory
   */
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
  free(result);
  return 0;
}

__global__ void kernel(float *a, float *b, float *result) {
  *result = 3;
}
