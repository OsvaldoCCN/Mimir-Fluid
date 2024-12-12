#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <thread>


// incluir Mimir para la visualizacion de la simulacion en GPU
#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda
using namespace std;

// Definición del tamaño del grid
#define NX 256
#define NY 256
#define BLOCK_SIZE 16  // Tamaño del bloque CUDA

__global__ void applyBoundaryConditions(float *u, float *v, float *p, float *d, int nx, int ny, int x1, int y1, int x2, int y2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j == 0 || j == ny - 1) {
        int idx = i + j * nx;

        // Velocidad cero en las paredes
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        p[idx] = 0.0f;
        d[idx] = 0.0f;


    }

        // Asegurarse de que el índice esté dentro del área del obstáculo
    if (i >= x1 && i < x2 && j >= y1 && j < y2) {
        int idx = i + j * nx;

        u[idx] = 0.0f;  // Velocidad en x dentro del obstáculo
        v[idx] = 0.0f;  // Velocidad en y dentro del obstáculo
        p[idx] = 0.0f;  // Presión dentro del obstáculo
        d[idx] = 0.0f;  // Divergencia dentro del obstáculo
    }
}


__device__ void applyVelocityLimit(float *u, float *v, int nx, int ny, float max_velocity) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = i + j * nx;

            // Calcular la magnitud de la velocidad
            float velocity_magnitude = sqrt(u[idx] * u[idx] + v[idx] * v[idx]);

            // Limitar la velocidad si supera el valor máximo
            if (velocity_magnitude > max_velocity) {
                float scale_factor = max_velocity / velocity_magnitude;
                u[idx] *= scale_factor;
                v[idx] *= scale_factor;
            }
        }
    }
}

__global__ void applyObstacle(float *u, float *v, float *p, float *d, int nx, int ny, int x1, int y1, int x2, int y2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Verificamos si el índice está dentro de las coordenadas del obstáculo
    if (i >= x1 && i < x2 && j >= y1 && j < y2) {
        int idx = i + j * nx;
        u[idx] = 0.0f;  // Velocidad en x dentro del obstáculo
        v[idx] = 0.0f;  // Velocidad en y dentro del obstáculo
        p[idx] = 0.0f;  // Presión dentro del obstáculo
        d[idx] = 0.0f;  // Divergencia dentro del obstáculo
    }

}

__global__ void computeDivergence(float *u, float *v, float *divergence, int nx, int ny, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i + j * nx;

        // Diferencias centradas para calcular la divergencia
        float du_dx = (u[idx + 1] - u[idx - 1]) / (2.0f * dx);
        float dv_dy = (v[idx + nx] - v[idx - nx]) / (2.0f * dy);

        divergence[idx] = du_dx + dv_dy;
    }
}



__global__ void solvePoisson(float *p, float *divergence, int nx, int ny, float dx, float dy, int maxIterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Variable para la convergencia
    float tolerance = 1e-5f;
    int iterations = 0;
    float error = tolerance + 1.0f;

    while (iterations < maxIterations && error > tolerance) {
        error = 0.0f;

        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            int idx = i + j * nx;

            // Cálculo de la presión usando diferencias centradas
            float p_x = (p[idx + 1] + p[idx - 1]) / (dx * dx);
            float p_y = (p[idx + nx] + p[idx - nx]) / (dy * dy);
            float newP = (p_x + p_y - divergence[idx]) / (2.0f / (dx * dx) + 2.0f / (dy * dy));

            error = fmaxf(error, fabsf(newP - p[idx]));  // Establecer el error como la diferencia máxima
            p[idx] = newP;
        }

        __syncthreads();  // Asegura que todos los threads actualicen antes de la siguiente iteración

        iterations++;
    }
}

__global__ void correctVelocities(float *u_new, float *v_new, float *p, int nx, int ny, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i + j * nx;

        // Gradientes de presión
        float dpdx = (p[idx + 1] - p[idx - 1]) / (2.0f * dx);
        float dpdy = (p[idx + nx] - p[idx - nx]) / (2.0f * dy);

        // Corrección de velocidades
        u_new[idx] -= dt * dpdx;
        v_new[idx] -= dt * dpdy;

        // Aplicar límite de velocidad  
        applyVelocityLimit(u_new, v_new, nx, ny, 1.0f);
    }
}

__global__ void computeVelocities(float *u, float *v, float *u_new, float *v_new, int nx, int ny, float dx, float dy, float dt, float nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Índice en x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Índice en y


    int idx = i + (j * nx); // Índice en el arreglo de la celda actual
    int idx_der = (i + 1) + (j * nx); // Índice en el arreglo de la celda derecha
    int idx_izq = (i - 1) + (j * nx); // Índice en el arreglo de la celda izquierda
    int idx_up = i + ((j + 1) * nx); // Índice en el arreglo de la celda de arriba
    int idx_down = i + ((j - 1) * nx); // Índice en el arreglo de la celda de abajo


    if (i > 0 && i < (nx - 1) && j > 0 && j < (ny - 1)) {
        // Gradientes de velocidad
        float du_dx = (u[idx_der] - u[idx_izq]) / (2.0f * dx);
        float du_dy = (u[idx_up] - u[idx_down]) / (2.0f * dy);
        float dv_dx = (v[idx_der] - v[idx_izq]) / (2.0f * dx);
        float dv_dy = (v[idx_up] - v[idx_down]) / (2.0f * dy);

        // Terminos de advección
        float u_dot_grad = u[idx] * du_dx + v[idx] * du_dy;
        float v_dot_grad = u[idx] * dv_dx + v[idx] * dv_dy;

        // Difusión
        float d2u_dx2 = (u[idx_der] - 2.0f * u[idx] + u[idx_izq]) / (dx * dx);
        float d2u_dy2 = (u[idx_up] - 2.0f * u[idx] + u[idx_down]) / (dy * dy);
        float d2v_dx2 = (v[idx_der] - 2.0f * v[idx] + v[idx_izq]) / (dx * dx);
        float d2v_dy2 = (v[idx_up] - 2.0f * v[idx] + v[idx_down]) / (dy * dy);

        // Termino de difusión
        float nu_laplacian_u = nu * (d2u_dx2 + d2u_dy2);
        float nu_laplacian_v = nu * (d2v_dx2 + d2v_dy2);


        // Ecuaciones de Navier-Stokes discretizadas
        u_new[idx] = u[idx] + dt * (
            - u_dot_grad
            + nu_laplacian_u
        );
        v_new[idx] = v[idx] + dt * (
            - v_dot_grad
            + nu_laplacian_v
        );
    
    }
}

__global__ void initialize(float *u, float *v, float *p, float *d, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int idx = i + j * nx;
        u[idx] = 0.0f;  // Inicializar velocidad en x
        v[idx] = 0.0f;  // Inicializar velocidad en y
        p[idx] = 0.0f;  // Inicializar presión
        d[idx] = 0.0f;  // Inicializar divergencia

        // solo asignar velocidad en la pared izquierda
        if (i == 0) {
            u[idx] = 1.0f;  
            v[idx] = 0.0f;

            //v[idx] = 0.2f;  // Velocidad en y en el centro de la pared izquierda
        }
    }



}

int main() {
    // Tamaño de la simulación

    // Parámetros de simulación
    /*
    float dx = 2.0f / (nx - 1); // Espaciado en x
    float dy = 2.0f / (ny - 1);
    float nu = 0.01f;  // Viscosidad
    float sigma = 0.2f;
    float dt = sigma * dx * dy / nu;  // Paso de tiempo
    */
    // Resolución de la grilla
    const int nx = 256;
    const int ny = 256;

    // Dominio físico (1m x 1m)
    const float domain_length = 1.0f; // en metros

    // Espaciado en x e y
    const float dx = domain_length / (nx - 1);
    const float dy = domain_length / (ny - 1);

    // Paso de tiempo ajustado para cumplir la condición CFL
    const float dt = 1.0f / 1000.0f;

    // Viscosidad cinemática (inicialmente mayor para estabilidad)
    const float nu = 0.01f; // En m^2/s



    // parametros
    size_t iter_count = 1000;
    unsigned long long seed = time(nullptr); // O cualquier otro método para generar un seed

    int x1 = nx / 2 - 10; // Esquina superior izquierda en x
    int y1 = ny / 2 - 60; // Esquina superior izquierda en y
    int x2 = nx / 2 + 10; // Esquina inferior derecha en x
    int y2 = ny / 2 + 60; // Esquina inferior derecha en y


    // Reservar memoria para los campos de velocidad en la CPU y la GPU
    float *u        = nullptr;
    float *v        = nullptr;
    float *u_new    = nullptr;
    float *v_new    = nullptr;
    float *divergence = nullptr;
    float *p        = nullptr;

    checkCuda(cudaMalloc((void**)&u, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&v, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&u_new, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&v_new, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&divergence, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&p, nx * ny * sizeof(float)));

    
    // Buffer para la visualización de la simulación
    MimirEngine engine;
    engine.init(1920, 1080);


    MemoryParams u1;
    u1.layout = DataLayout::Layout2D;
    u1.element_count = {(unsigned)(nx), (unsigned)(ny)};
    u1.component_type = ComponentType::Float;
    u1.channel_count = 1;// copilot dijo que era 2
    u1.resource_type = ResourceType::Buffer;
    auto points_u = engine.createBuffer((void**)&u, u1);

    ViewParams view_u1;
    view_u1.element_count = nx * ny;
    view_u1.extent = {(unsigned)(nx), (unsigned)(ny),1};
    view_u1.data_domain = DataDomain::Domain2D;
    view_u1.domain_type = DomainType::Structured;
    view_u1.view_type = ViewType::Voxels; // o puede ser
    view_u1.attributes[AttributeType::Color] = *points_u;
    view_u1.options.default_color = {256,0,0}; // no sirve segun Isaias
    view_u1.options.default_size = 1;

    engine.createView(view_u1);

MemoryParams p1;
    p1.layout = DataLayout::Layout2D;
    p1.element_count = {(unsigned)(nx), (unsigned)(ny)};
    p1.component_type = ComponentType::Float;
    p1.channel_count = 1;// copilot dijo que era 2
    p1.resource_type = ResourceType::Buffer;
    auto points = engine.createBuffer((void**)&p, p1);

    ViewParams view_p1;
    view_p1.element_count = nx * ny;
    view_p1.extent = {(unsigned)(nx), (unsigned)(ny),1};
    view_p1.data_domain = DataDomain::Domain2D;
    view_p1.domain_type = DomainType::Structured;
    view_p1.view_type = ViewType::Voxels; // o puede ser
    view_p1.attributes[AttributeType::Color] = *points;
    view_p1.options.default_color = {256,0,0}; // no sirve segun Isaias
    view_p1.options.default_size = 1;

    engine.createView(view_u1);

    engine.displayAsync();

    checkCuda(cudaDeviceSynchronize());

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cout << "Inicializando..." << endl;
    // Inicializar campos de velocidad
    initialize<<<numBlocks, threadsPerBlock>>>(u, v, p, divergence, nx, ny);

    cout << "Simulando..." << endl;
    for(size_t i = 0; i < iter_count; i++){

        (i % 10 == 0)?cout << "Iteración " << i << endl: cout << "";

        // Calcular la divergencia
        computeDivergence<<<numBlocks, threadsPerBlock>>>( u, v, divergence, nx, ny, dx, dy);
        checkCuda(cudaDeviceSynchronize());

        // Calcular la presión
        solvePoisson<<<numBlocks, threadsPerBlock>>>(p, divergence, nx, ny, dx, dy, 100);
        checkCuda(cudaDeviceSynchronize());      

        // Calcular las velocidades en el siguiente paso de tiempo
        computeVelocities<<<numBlocks, threadsPerBlock>>>(u, v, u_new, v_new, nx, ny, dx, dy, dt, nu);
        checkCuda(cudaDeviceSynchronize());

        // Corregir las velocidades
        correctVelocities<<<numBlocks, threadsPerBlock>>>(u_new, v_new, p, nx, ny, dx, dy, dt);
        checkCuda(cudaDeviceSynchronize());

        // Aplicar el obstáculo
        applyObstacle<<<numBlocks, threadsPerBlock>>>(u_new, v_new, p, divergence, nx, ny, x1, y1, x2, y2);
        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaMemcpy(u, u_new, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice));
        checkCuda(cudaMemcpy(v, v_new, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice));

        // Aplicar condiciones de frontera
        applyBoundaryConditions<<<numBlocks, threadsPerBlock>>>(u, v, p, divergence, nx, ny, x1, y1, x2, y2);
        checkCuda(cudaDeviceSynchronize());


        engine.updateViews();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    engine.showMetrics();
    engine.exit();


    checkCuda(cudaFree(u));
    checkCuda(cudaFree(v));
    checkCuda(cudaFree(u_new));
    checkCuda(cudaFree(v_new));
    checkCuda(cudaFree(divergence));
    checkCuda(cudaFree(p));


    return 0;
}

