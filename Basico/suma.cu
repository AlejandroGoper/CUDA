#include <iostream>

using namespace std;

/*
    Definición del Kernel 
*/
__global__ void sumar(int *a, int *b, int *r){
    *r = *a + *b;
}


int main(int argc, char** argv){

    int a = 11, b = 11, c = 0; // Copias del Host
    int *d_a,*d_b,*d_c; // Copias del Device

    //Reservo memoria en la GPU
    cudaMalloc((void **)&d_a,sizeof(int));
    cudaMalloc((void **)&d_b,sizeof(int));
    cudaMalloc((void **)&d_c,sizeof(int));

    //Copio valores del Host al Device
    cudaMemcpy(d_a,&a,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,sizeof(int),cudaMemcpyHostToDevice);

    //Lanzo el kernel

    sumar<<<1,1>>>(d_a,d_b,d_c);

    //Copio el resultado de Device a Host
    cudaMemcpy(&c,d_c,sizeof(int),cudaMemcpyDeviceToHost);

    cout << c << endl; 

    // Lipio variables de GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
