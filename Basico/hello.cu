#include <stdio.h> 
#define N 1024

__global__ void vectorAdd(int *a, int *b, int *c){
	int i= threadIdx.x;
	*(c+i) = *(a+i) + *(b+i);
}

int main(void)
{
	int *a,*b,*c;
	cudaMallocManaged(&a,N*sizeof(int));
	cudaMallocManaged(&b,N*sizeof(int));
	cudaMallocManaged(&c,N*sizeof(int));
	for(int i = 0; i < N; ++i){
		*(a+i) = 2*i;
		*(b+i) = i;
		*(c+i) = 0;
	}

	vectorAdd<<<1,N>>>(a,b,c);

	cudaDeviceSynchronize();
	
	for(int i = 0; i < 10 ; ++i){
		printf("c[%d] = %d\n",i,c[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return 0;
}
