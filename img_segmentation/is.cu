#include "is.h"
#include <cuda_runtime.h>
#include <iostream>

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


#define NPP_MAXABS_32F ( 3.402823466e+38f )
#define DIMX 32
#define DIMY 4
#define THREADS (DIMX*DIMY)

// each block considers all posible positions for a rectangle of size `(blockIdx.x + 1) X (blockIdx.y + 1)`
__global__ void compute(const float* d, float* r, int* l,
                        const int nx, const int ny, const int nnx)
{
    const int wx = blockIdx.x + 1;
    const int wy = blockIdx.y + 1;

    const int size = nx*ny;
    const int size_rect = wy*wx;
    const int size_bg = size - size_rect;

    if (sizex >= size)
        return;

    // location of best found position, (y0, x0, y1, x1)
    __shared__ int loc[THREADS*4];
    // parameters to compute the error at the best found position, (score, background error, rectangle error)
    __shared__ float res[THREADS*3];

    const int id = threadIdx.x + threadIdx.y*blockDim.x;

    const float va = d[nx + ny*nnx];
    const float vas = -2*size_rect*va;

    res[3*id] = -NPP_MAXABS_32F;

    int x, y;

    for (y = 0; y <= ny - wy; y += blockDim.y)
    {
        const int y0 = y + threadIdx.y;
        const int y1 = y0 + wy;

        for (x = 0; x <= nx - wx; x += blockDim.x)
        {
            const int x0 = x + threadIdx.x;
            const int x1 = x0 + wx;

            if (x1 > nx || y1 > ny)
                continue;



            float vx = d[x1 + y1*nnx];
            vx -= d[x0 + y1*nnx];
            vx -= d[x1 + y0*nnx];
            vx += d[x0 + y0*nnx];

            const float cx = vx;

            vx *= vx;
            vx *= size;
            vx += vas*cx;


            if (vx > res[3*id])
            {
                res[3*id + 0] = vx;
                res[3*id + 1] = va - cx;
                res[3*id + 2] = cx;

                loc[4*id + 0] = y0; loc[4*id + 1] = x0;
                loc[4*id + 2] = y1; loc[4*id + 3] = x1;
            }
        }
    }


    __syncthreads();

    if (!id)
    {
        for (int i = 1; i < THREADS; i++)
        {
            if (res[3*i] > res[0])
            {
                res[0] = res[i*3];
                res[1] = res[i*3 + 1];
                res[2] = res[i*3 + 2];

                loc[0] = loc[i*4 + 0]; loc[1] = loc[i*4 + 1];
                loc[2] = loc[i*4 + 2]; loc[3] = loc[i*4 + 3];
            }
        }

        const float max = (res[1] * res[1]) / size_bg
            + (res[2] * res[2]) / size_rect;

        const int bid = blockIdx.x + blockIdx.y*nx;

        r[3*bid + 0] = max;
        r[3*bid + 1] = res[1] / size_bg;
        r[3*bid + 2] = res[2] / size_rect;

        l[4*bid + 0] = loc[0];
        l[4*bid + 1] = loc[1];
        l[4*bid + 2] = loc[2];
        l[4*bid + 3] = loc[3];
    }
}

/* segment 1-bit image to a rectangle and background such that the sum of squared errors is minimized */
Result segment(int ny, int nx, const float* data) {
    const int nnx = nx + 1;
    const int nny = ny + 1;

    float* pre = (float*) std::malloc(nny * nnx * sizeof(float));

    int x, y;

    for (x = 0; x < nnx; x++)
        pre[x] = 0.0f;

    // precompute scan for each row
    for (y = 0; y < ny; y++)
    {
        const int y0 = y + 1;
        pre[y0*nnx] = 0.0f;
        float sum = 0.0f;

        for (x = 0; x < nx; x++)
        {
            const int x0 = x + 1;
            sum += data[3*x + 3*nx*y];
            pre[x0 + y0*nnx] += sum;
        }
    }

    dim3 dimBlock(DIMX, DIMY);
    dim3 dimGrid(nx, ny);

    float* dGPU = NULL;
    float* rGPU = NULL;
    int* lGPU = NULL;

    int* l = (int*) std::malloc((nx*ny - 1) * 4 * sizeof(int));
    float* r = (float*) std::malloc((nx*ny - 1) * 3 * sizeof(float));

    CHECK(cudaMalloc((void**) &dGPU, nny * nnx * sizeof(float)));
    CHECK(cudaMalloc((void**) &rGPU, (nx*ny - 1) * 3 * sizeof(float)));
    CHECK(cudaMalloc((void**) &lGPU, (nx*ny - 1) * 4 * sizeof(int)));

    CHECK(cudaMemcpy(dGPU, pre, nny * nnx * sizeof(float), cudaMemcpyHostToDevice));

    compute<<<dimGrid, dimBlock>>>(dGPU, rGPU, lGPU, nx, ny, nnx);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(l, lGPU, (nx*ny - 1) * 4 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(r, rGPU, (nx*ny - 1) * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    float m = r[0];
    float c[2] = {r[1], r[2]};
    int ll[4] = {l[0], l[1], l[2], l[3]};

    // find the best solution from the results
    for (int i = 1; i < nx*ny - 1; i++)
    {
        if (r[3*i] > m)
        {
            m = r[3*i];
            c[0] = r[3*i + 1];
            c[1] = r[3*i + 2];

            ll[0] = l[4*i + 0]; ll[1] = l[4*i + 1];
            ll[2] = l[4*i + 2]; ll[3] = l[4*i + 3];
        }
    }

    Result result = {
        ll[0],
        ll[1],
        ll[2],
        ll[3],
        {c[0], c[0], c[0]},
        {c[1], c[1], c[1]},
    };

    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    CHECK(cudaFree(lGPU));
    std::free(pre);
    std::free(r);
    std::free(l);

    return result;
}
