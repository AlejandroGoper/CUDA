#include <iostream>

using namespace std;

// Defino el kernel de sumar matrices
__global__ void suma_gpu(int *A, int *B, int *R){

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;

    R[i][j] = A[i][j] + B[i][j];
}



// FUncion en la CPU
void M_random_int(int **, int);    
void suma_cpu(int **, int **, int**,int);

void imprimir_matriz(int **, int);

int main(){
    int N = 16;
    //Host
    int **A, **B,**C,**R;
    // Device 
    int **A_gpu, **B_gpu, **C_gpu;

    // Reservo memoria en la CPU
    A = new int*[N];
    B = new int*[N];
    C = new int*[N];
    R = new int*[N];
    for(int k = 0; k<N; k++){
        *(A+k) = new int[N];
        *(B+k) = new int[N];
        *(C+k) = new int[N];
        *(R+k) = new int[N];
    }
    // Lleno matriz A y B en la CPU
    M_random_int(A,N);
    M_random_int(B,N);
    // IMprimo en CPU
    cout << "A" << endl;
    imprimir_matriz(A,3);
    cout << "B" << endl;
    imprimir_matriz(B,3);
    //suma_cpu(A,B,C,N);
    //cout << "C" << endl;
    //imprimir_matriz(C,3);


    // Reservo memoria en la GPU 
    cudaMalloc(&A_gpu,N*sizeof(int *)); // Analogo a new int*[N];
    cudaMalloc(&B_gpu,N*sizeof(int *)); // Analogo a new int*[N];
    cudaMalloc(&C_gpu,N*sizeof(int *)); // Analogo a new int*[N];
    for(int k = 0; k < N ; k++){
        cudaMalloc(&(A_gpu[k]),N*sizeof(int));
        cudaMalloc(&(B_gpu[k]),N*sizeof(int));
        cudaMalloc(&(C_gpu[k]),N*sizeof(int));
    }
    // Copiamos a la memoria global
    cudaMemcpy(A_gpu,A,N*sizeof(int *),cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu,B,N*sizeof(int *),cudaMemcpyHostToDevice);
    for(int k = 0; k < N ; k++){
        // De una vez copiamos valores de CPU a GPU a la memoria global 
        cudaMemcpy(A_gpu[k],A[k],N*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(B_gpu[k],B[k],N*sizeof(int),cudaMemcpyHostToDevice);
    }

    // Lanzamos el KERNEL con 1 bloque, 16*16 hilos
    dim3 grid(1,1,1); 
    dim3 hilos_por_bloque(N,N,1);
    suma_gpu<<<grid,hilos_por_bloque>>>(A_gpu,B_gpu,C_gpu);
   
    cudaMemcpy(C,C_gpu,N*sizeof(int *),cudaMemcpyDeviceToHost);
    //Copio el resultado de Device a Host 
    for(int k = 0; k < N; k++){
        cudaMemcpy(C[k],C_gpu[k],N*sizeof(int),cudaMemcpyDeviceToHost);
    }
    //cudaMemcpy(C,C_gpu,N*sizeof(int *),cudaMemcpyDeviceToHost);

    cout << "C_gpu" << endl;
    imprimir_matriz(C_gpu,3);

    return 0;
}

void M_random_int(int **r, int n){    
    for(int i = 0; i<n ; i++){
        for(int j = 0; j < n; j++){
            *(*(r+i)+j) = rand()%5;
        }
    }
}
void imprimir_matriz(int **r, int n){    
    for(int i = 0; i<n ; i++){
        for(int j = 0; j < n; j++){
            cout << r[i][j] << " ";
        }
        cout << endl;
    }
}



void suma_cpu(int **A, int **B, int **R,int N){
    for(int i = 0; i<N; i++){
        for(int j = 0; j < N; j++){
            R[i][j] = A[i][j] + B[i][j];
        }
    }
}

