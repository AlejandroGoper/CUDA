#include <iostream>
//#include <cstdlib>

using namespace std;

/*
    Definición del Kernel 
*/
__global__ void sumar(int *a, int *b, int *r){
    int i = threadIdx.x;
    *(r+i) = *(a+i) + *(b+i);
}

void random_ints(int *r, int n){
    
    for(int i = 0; i<n ; i++){
        *(r+i) = rand()%5000;
    }
}

int main(int argc, char** argv){
    int N = 10;
    int *a, *b, *c; // Copias del Host
    int *d_a,*d_b,*d_c; // Copias del Device

    //Reservo memoria en la CPU
    a = new int[N];
    b = new int[N];
    c = new int[N];
     
    //Lleno a y b de valores aleatorios.
    random_ints(a,N);
    random_ints(b,N);

    //Imprimiendo 
    for(int i = 0; i<N; i++){
        cout << *(a+i) << "\t" << *(b+i) << endl;
    }

    int size = N*sizeof(int);

    //Reservo memoria en la GPU
    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);

    //Copio valores del Host al Device
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);

    //Lanzo el kernel

    sumar<<<1,N>>>(d_a,d_b,d_c);

    //Copio el resultado de Device a Host
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

    cout << endl;
    for(int i = 0; i < N; i++){
        cout << *(c+i) << endl;
    } 

    //Limpio variables de CPU
    delete [] a;
    delete [] b;
    delete [] c;
 
    // Lipio variables de GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}