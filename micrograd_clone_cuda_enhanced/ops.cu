#include <cuda_runtime.h>
#include <stdio.h>
#include "math.h"


extern "C" float * move_to_gpu(float * a, int sz){
    float * d_a;
    cudaMalloc((void**) &d_a, sz);
    cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
    return d_a;
}


extern "C" void move_to_cpu(float * a, float * d_a, int sz){
    cudaMemcpy(a, d_a, sz, cudaMemcpyDeviceToHost);
}


extern "C" float * allocate_on_gpu(int sz){
    float * a;
    cudaMalloc((void **) &a, sz);


    return a;
}


__global__ void matrix_zeros_kernel(float * mat_a, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (x < a_1 && y < a_2){
        mat_a[a_1 * x + y] = 0;
    }

    
}   


extern "C" float * zeros_gpu(int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    matrix_zeros_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_out, a_1, a_2);


    return mat_out;
}

__global__ void matrix_transpose_kernel(float * mat_a, float * mat_out,  int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (x < a_1 && y < a_2){
        mat_out[y * a_2 + x] = mat_a[a_1 * x + y];
    }

    
}   


extern "C" float * transpose_gpu(float * mat_a, int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_out, a_1, a_2);


    return mat_out;
}



__global__ void matrix_ones_kernel(float * mat_a, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (x < a_1 && y < a_2){
        mat_a[a_2 * x + y] = 1;
    }

    
}   


extern "C" float * ones_gpu(int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    matrix_ones_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_out, a_1, a_2);


    return mat_out;
}




__global__ void matmul_kernel(float * mat_a, float * mat_b, float * mat_out, int a_1, int a_2, int b_2){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < a_1 && col < b_2){
        float sum = 0;
        for (int i = 0; i < a_2; i++){
            sum += (mat_a[a_2 * row + i] * mat_b[b_2 * i + col]);  
        }

        mat_out[b_2 * row + col] = sum;
    }

    
}   


extern "C" float * matmul_gpu(float * mat_a, float * mat_b, int a_1, int a_2, int b_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * b_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (b_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);
    
    // printf("%d, %d", blocksPerGrid.x, blocksPerGrid.y);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_b, mat_out, a_1, a_2, b_2);
    
    cudaDeviceSynchronize();

    return mat_out;
}



__global__ void add_kernel(float * mat_a, float * mat_b, float * mat_out, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = (mat_a[a_2 * x + y] + mat_b[a_2 * x + y]);
    }


}

extern "C" float * add_gpu(float * mat_a, float * mat_b, int a_1, int a_2){
    float * mat_out;
        mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_b, mat_out, a_1, a_2);
    cudaDeviceSynchronize();
    
    return mat_out;
}

__global__ void mul_kernel(float * mat_a, float * mat_b, float * mat_out, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = (mat_a[a_2 * x + y] * mat_b[a_2 * x + y]);
    }


}

extern "C" float * mul_gpu(float * mat_a, float * mat_b, int a_1, int a_2){
    float * mat_out;
        mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_b, mat_out, a_1, a_2);

    
    return mat_out;
}



__global__ void scalar_mul_kernel(float * mat_a, float * mat_out, float val, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = (mat_a[a_2 * x + y] * val);
    }
}

extern "C" float * scalar_mul_gpu(float * mat_a, float val, int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    scalar_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_out, val, a_1, a_2);

    
    return mat_out;
}


__global__ void tanh_kernel(float * mat_a, float * mat_out, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = tanh(mat_a[a_2 * x + y]);
    }


}

extern "C" float * tanh_gpu(float * mat_a, int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    tanh_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_out, a_1, a_2);


    return mat_out;
}


__global__ void relu_kernel(float * mat_a, float * mat_out, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = mat_a[a_2 * x + y] > 0 ? mat_a[a_2 * x + y] : 0;
    }


}

extern "C" float * relu_gpu(float * mat_a, int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_out, a_1, a_2);


    return mat_out;
}

__global__ void relugrad_kernel(float * mat_a, float * mat_out, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = mat_a[a_2 * x + y] > 0 ? 1 : 0;
    }


}

extern "C" float * relugrad_gpu(float * mat_a, int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_out, a_1, a_2);


    return mat_out;
}


__global__ void pow_kernel(float * mat_a, float * mat_out, float val, int a_1, int a_2){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < a_1 && y < a_2){
        mat_out[a_2 * x + y] = pow (mat_a[a_2 * x + y], val) ;
    }


}

extern "C" float * pow_gpu(float * mat_a, float val, int a_1, int a_2){
    float * mat_out;
    mat_out = allocate_on_gpu(a_1 * a_2 * sizeof(float));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((a_1 + threadsPerBlock.x  - 1) / threadsPerBlock.x, 
                      (a_2 + threadsPerBlock.y  - 1) / threadsPerBlock.y);

    pow_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_a, mat_out, val, a_1, a_2);


    return mat_out;
}

