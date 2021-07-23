/*
    Nombre derivado de: Matrix multiplication using global memory

    Autor: Alejandro Gómez
    Basado en: http://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/CUDA2D/CUDA2D.html

    El codigo realiza la multiplicacion de dos matrices cuadradas utilizando
    la memoria global y bloques multidimensionales.
*/

/*
    ==========================================================================================
        Declaración del kernel Bidimensional
    ==========================================================================================
*/

__global__ void kernel(float *A_row, float *B_col,float *R, int dim){
    // Calculamos el indice de la columa del elemento de R, denotado por x
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    // Caculamos el indice de la fila del elemento de R, denotado por y
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    // Para realizar la sumatoria de los productos de la fila de A por la columna de B
    float Rvalor = 0;
    // cada hilo calcula un valor de R 
    for(int k=0; k < dim ; k++){
        Rvalor+=A_row[y*dim+k]*B_col[k*dim+x];
    }
    // lo escribimos en la memoria global
    R[y*dim + x] = Rvalor;
}
