#include <stdio.h>
#include <stdlib.h>


/*
    ====================================================================
    Este programa es para realizar una suma de matrices cuadradas A y B
    de dimension nxn, con el uso de programacion paralela
    
    - Cada hilo ejecutara un elemento de la matriz de resultado S 

    A = [[ 1, 2, 4
           3, 7, 9
           7, 1, 2 ]]
    
    B = [[ 3, 8, 1
           6, 5, 4
           7, 0, 3 ]]

    Se utilizaran las siguientes matrices, puestas en archivos externos.
    A.dat y B.dat

    Autor: Alejandro Gómez.
    ====================================================================
*/

/* ::::::::::::::: KERNEL ::::::::::::::::::::::
    Argumentos: 
    - A es la matriz 1
    - B es la matiz 2 
    - S sera la suma de A+B
    - n es la dimension
   :::::::::::::::::::::::::::::::::::::::::::::
*/ 
__global__ void add_matrix(float *A, float *B,float *S, int n) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if(index < n*n){
        S[index] = A[index] + B[index];
    }
}

// recibe como parametros un caracter para saber si es la matriz A o B y devuelve a la matriz
void leer_matriz(char, float *);
void imprimir_matriz(float*,int n);



int main(int argc, char *argv[]){


    // Variables para el host
    float *A,*B, *S;     
    int n, size;
    
    // Variables para el device
    float *d_A, *d_B, *d_S;

    // Se guardaran las matrices en un solo array de dimension n*n en lugar de array de arrays

    printf("Dimension de las matrices cuadradas:");
    scanf("%d",&n);
    
    // ::::::::::: Reservando memoria en HOST ::::::::::::::::::::::
    
    size = n*n*sizeof(float);
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    S = (float*)malloc(size);
   
    leer_matriz('A',A);
    leer_matriz('B',B);

    printf("Imprimiendo A:\n");
    imprimir_matriz(A,n);
    printf("Imprimiendo B:\n");
    imprimir_matriz(B,n);

    // :::::::::: Reservando memoria en DEVICE :::::::::::::::::::::::::


    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_S,size);

    // ::::::::: Copiando de HOST a DEVICE :::::::::::::::::::::::::::::
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    // ::::::::: Lanzando KERNEL :::::::::::::::::::::::::::::::::::::::
    
    // Lanzo kernel con 1 bloque de 9 hilos
    add_matrix<<<1,9>>>(d_A,d_B,d_S,n);

    // ::::::::::::::::::: Copiando de DEVICE a HOST :::::::::::::::::::
    cudaMemcpy(S,d_S,size,cudaMemcpyDeviceToHost);
    printf("Imprimiendo resultado: \n");
    imprimir_matriz(S,n);

    // :::::::::::::::::: Liberando espacio en GPU :::::::::::::::::::::
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_S);

    // :::::::::::: Imprimiendo resutados de GPU :::::::::::::::::::::::::
  


    return 0;
}

void leer_matriz(char c, float *M){

    FILE *fp;
    if(c == 'A'){
        fp = fopen("A.dat","r");
        int i = 0;    
        while(feof(fp)== 0){
            fscanf(fp,"%f",&M[i]);
            i++;
        }
        /*for(int k = 0; k<3; k++){
            for(int l= 0; l < 3; l++){
                int index = k*3 + l;
                fscanf(fp,"%f",&M[index]);
            }
        }*/
        fclose(fp);
    }
    else if(c == 'B'){
        fp = fopen("B.dat","r");
        int i = 0;   
        while(feof(fp)== 0){
            fscanf(fp,"%f",&M[i]);
            i++;
        }
        fclose(fp);
    }
    else{
        printf("Caracter no reconocido correctamente.");
    }
}

void imprimir_matriz(float *M, int n){
    int i,j,index;
    for(i = 0; i<n; i++){
        for(j=0; j<n; j++){
            index = i*3 + j;
            printf("%.1f ", M[index]);
        }
        printf("\n");
    }
}