#include <iostream>
//#include <cstdlib>

using namespace std;

/*
    Para entender como se indexan los elementos por bloque y por hilo, revisar 
    la siguiente liga: https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

    Definición del Kernel:
    
    Si cada bloque tiene M hilos, el índice correspondiente será:
    id = threadIdx.x + M*blockIdx.x;

    Pero: M = blockDim.x;

    Solo podemos lanzar maximo 1024 hilos por bloque 
    pero podemos lanzar 2^32 - 1 bloques.
*/
__global__ void sumar(int *a, int *b, int *r){
    int M = blockDim.x;
    int i = threadIdx.x + M*blockIdx.x;
    *(r+i) = *(a+i) + *(b+i);
}

void random_ints(int *r, int n){
    
    for(int i = 0; i<n ; i++){
        *(r+i) = rand()%5000;
    }
}

int main(int argc, char** argv){
    int N = 10000; //longitud del arreglo
    int hilos_por_bloque = 512;
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
    for(int i = N; i>=N-10; i--){
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
    int num_bloques = N / hilos_por_bloque;
    sumar<<<num_bloques+1,hilos_por_bloque>>>(d_a,d_b,d_c);

    //Copio el resultado de Device a Host
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

    cout << endl;
    for(int i = 0; i <= 10; i++){
        cout << *(c+i) << endl;
    } 

    cout <<"\t\t" << c[N-1]<< endl;

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
