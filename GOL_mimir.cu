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
#define BLOCK_SIZE 16  // Tamaño del bloque CUDA

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

__global__ void applyObstacle(float *u, float *v, float *p, int nx, int ny, int x1, int y1, int x2, int y2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Verificamos si el índice está dentro de las coordenadas del obstáculo
    if (i >= x1 && i < x2 && j >= y1 && j < y2) {
        int idx = i + j * nx;
        u[idx] = 0.0f;  // Velocidad en x dentro del obstáculo
        v[idx] = 0.0f;  // Velocidad en y dentro del obstáculo
        p[idx] = 0.0f;  // Presión dentro del obstáculo
    }

}


__global__ void computeDivergence(float *divergence, float *u, float *v, int nx, int ny, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i + j * nx;
        float du_dx = (u[i + 1 + j * nx] - u[i - 1 + j * nx]) / (2.0f * dx);
        float dv_dy = (v[i + (j + 1) * nx] - v[i + (j - 1) * nx]) / (2.0f * dy);
        divergence[idx] = du_dx + dv_dy;
    }
}


__global__ void solvePressurePoisson(float *p, float *divergence, int nx, int ny, float dx, float dy, int iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int it = 0; it < iterations; ++it) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            int idx = i + j * nx;
            float p_x = (p[i + 1 + j * nx] + p[i - 1 + j * nx]) / (dx * dx);
            float p_y = (p[i + (j + 1) * nx] + p[i + (j - 1) * nx]) / (dy * dy);
            p[idx] = (p_x + p_y - divergence[idx]) / (2.0f / (dx * dx) + 2.0f / (dy * dy));
        }
        __syncthreads(); // Asegura que todos los hilos actualicen antes de la siguiente iteración
    }
}

__global__ void computeVelocities(float *u, float *v, float *u_new, float *p, float *v_new, int nx, int ny, float dx, float dy, float dt, float nu, float rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Índice en x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Índice en y


    /*
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1) {
        u_new[idx] = 0.0f;  // Sin movimiento en las paredes en la dirección x
        v_new[idx] = 0.0f;  // Sin movimiento en las paredes en la dirección y
    }
*/

    if (i < (nx - 1) && j < (ny - 1)) {

        int idx = i + (j * nx); // Índice en el arreglo de la celda actual
        int idx_der = (i + 1) + (j * nx); // Índice en el arreglo de la celda derecha
        int idx_izq = (i - 1) + (j * nx); // Índice en el arreglo de la celda izquierda
        int idx_up = i + ((j + 1) * nx); // Índice en el arreglo de la celda de arriba
        int idx_down = i + ((j - 1) * nx); // Índice en el arreglo de la celda de abajo

        // Divergencia de la velocidad
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


        // Presión
        float dp_dx = (p[idx_der] - p[idx_izq]) / (2.0f * dx);
        float dp_dy = (p[idx_up] - p[idx_down]) / (2.0f * dy);

        // Termino de presión
        float pressure_term_x = dp_dx / rho;
        float pressure_term_y = dp_dy / rho;



        // Ecuaciones de Navier-Stokes discretizadas
        u_new[idx] = u[idx] - dt * 
        (u_dot_grad +       // Advección
        pressure_term_x -   // Presión
        nu_laplacian_u);    // difusión

        v_new[idx] = v[idx] - dt *
        (v_dot_grad +       // Advección
        pressure_term_y -   // Presión
        nu_laplacian_v);    // Difusión

        // Aplicar límite de velocidad  
        //applyVelocityLimit(u_new, v_new, nx, ny, 1.0f);
    }

}


__global__ void calculateVorticity(float *u, float *v, float *vorticity, int nx, int ny, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Comprobamos que estamos dentro de los límites de la malla
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        // Derivadas parciales de u y v
        float du_dy = (u[(i + (j + 1) * nx)] - u[(i + (j - 1) * nx)]) / (2.0f * dy);  // Derivada parcial de u con respecto a y
        float dv_dx = (v[(i + 1 + j * nx)] - v[(i - 1 + j * nx)]) / (2.0f * dx);  // Derivada parcial de v con respecto a x

        // La vorticidad es la diferencia entre las derivadas parciales
        vorticity[i + j * nx] = dv_dx - du_dy;
    }
}


__global__ void calculateVelocityMagnitude(float *u, float *v, float *magnitude, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int idx = i + j * nx;
        magnitude[idx] = sqrtf(u[idx] * u[idx] + v[idx] * v[idx]);
    }
}

__global__ void initialize(float *u, float *v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny ) {
        int idx = i + j * nx;
        u[idx] = 0.0f;  // Inicializar velocidad en x
        v[idx] = 0.0f;  // Inicializar velocidad en y

        // solo asignar velocidad en la pared izquierda
        if (i == 0) {
            u[idx] = 1.0f;  
            //v[idx] = 0.01f;  // Velocidad en y en el centro de la pared izquierda
        }

    }

}

int main() {


    // Parámetros de simulación
    /*
    float dx = 2.0f / (nx - 1); // Espaciado en x
    float dy = 2.0f / (ny - 1);
    float nu = 0.01f;  // Viscosidad
    float sigma = 0.2f;
    float dt = sigma * dx * dy / nu;  // Paso de tiempo
    */
   // simular agua
    // Tamaño de la simulación
    int nx = 256;        // Número de celdas en x
    int ny = 256;        // Número de celdas en y
    float dx = 0.001f;   // Tamaño de celda en x (1 mm)
    float dy = 0.001f;   // Tamaño de celda en y (1 mm)
    float nu = 1e-6f;    // Viscosidad cinemática para agua (m²/s)
    float dt = 0.0005f;  // Paso de tiempo (s) para estabilidad según CFL

    // parametros
    size_t iter_count = 1000;
    unsigned long long seed = time(nullptr); // O cualquier otro método para generar un seed

    int x1 = nx / 2 - 10; // limite pared izquierda en x
    int y1 = ny / 2 - 60; // limite pared izquierda en y 
    int x2 = nx / 2 ; // limite pared derecha en x
    int y2 = ny / 2 + 60; // limite pared derecha en y

    


    // Reservar memoria para los campos de velocidad en la CPU y la GPU
    float *u            = nullptr;  checkCuda(cudaMalloc((void**)&u, nx * ny * sizeof(float)));
    float *v            = nullptr;  checkCuda(cudaMalloc((void**)&v, nx * ny * sizeof(float)));
    float *u_new        = nullptr;  checkCuda(cudaMalloc((void**)&u_new, nx * ny * sizeof(float)));
    float *v_new        = nullptr;  checkCuda(cudaMalloc((void**)&v_new, nx * ny * sizeof(float)));
    float *p            = nullptr;  checkCuda(cudaMalloc((void**)&p, nx * ny * sizeof(float)));
    float *divergence   = nullptr;  checkCuda(cudaMalloc((void**)&divergence, nx * ny * sizeof(float)));
    float *magnitude    = nullptr;  checkCuda(cudaMalloc((void**)&magnitude, nx * ny * sizeof(float)));


    // Buffer para la visualización de la simulación
    MimirEngine engine;
    engine.init(1920, 1080);


    MemoryParams m1;
    m1.layout = DataLayout::Layout2D;
    m1.element_count = {(unsigned)(nx), (unsigned)(ny)};
    m1.component_type = ComponentType::Float;
    m1.channel_count = 1;
    m1.resource_type = ResourceType::Buffer;
    auto points = engine.createBuffer((void**)&u, m1);

    ViewParams p1;
    p1.element_count = nx * ny;
    p1.extent = {(unsigned)(nx), (unsigned)(ny),1};
    p1.data_domain = DataDomain::Domain2D;
    p1.domain_type = DomainType::Structured;
    p1.view_type = ViewType::Voxels;
    p1.attributes[AttributeType::Color] = *points;
    p1.options.default_color = {256,0,0};
    p1.options.default_size = 1;

    engine.createView(p1);


    MemoryParams m_presion;
    m_presion.layout = DataLayout::Layout2D;
    m_presion.element_count = {(unsigned)(nx), (unsigned)(ny)};
    m_presion.component_type = ComponentType::Float;
    m_presion.channel_count = 1;
    m_presion.resource_type = ResourceType::Buffer;
    auto presion = engine.createBuffer((void**)&p, m_presion);

    ViewParams p_presion;
    p_presion.element_count = nx * ny;
    p_presion.extent = {(unsigned)(nx), (unsigned)(ny),1};
    p_presion.data_domain = DataDomain::Domain2D;
    p_presion.domain_type = DomainType::Structured;
    p_presion.view_type = ViewType::Voxels; // o puede ser
    p_presion.attributes[AttributeType::Color] = *presion;
    p_presion.options.default_color = {0,0,256}; // no sirve segun Isaias
    p_presion.options.default_size = 1;

    engine.createView(p_presion);
    


    engine.displayAsync();

    checkCuda(cudaDeviceSynchronize());

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cout << "Inicializando..." << endl;
    // Inicializar campos de velocidad
    initialize<<<numBlocks, threadsPerBlock>>>(u, v, nx, ny);

    cout << "Simulando..." << endl;
    for(size_t i = 0; i < iter_count; i++){
        (i%10==0? cout << "Iteración: " << i << endl : cout << "");
        // Calcular las velocidades en el siguiente paso de tiempo
        computeVelocities<<<numBlocks, threadsPerBlock>>>(u, v, p, u_new, v_new, nx, ny, dx, dy, dt, nu, 1.0f);
        checkCuda(cudaDeviceSynchronize());
        
        checkCuda(cudaMemcpy(u, u_new, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice)); // Copia el resultado de u_new a u
        checkCuda(cudaMemcpy(v, v_new, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice)); // Copia el resultado de v_new a v
        checkCuda(cudaDeviceSynchronize());
        
        // Buscar otra manera de intercambiar punteros, mejor despues de esto, trabajar con u_new y v_new
    


        // Calcular la magnitud de la velocidad
        //calculateVelocityMagnitude<<<numBlocks, threadsPerBlock>>>(u, v, magnitude, nx, ny);
        //checkCuda(cudaDeviceSynchronize());


        // Calcular la divergencia
        computeDivergence<<<numBlocks, threadsPerBlock>>>(divergence, u_new, v_new, nx, ny, dx, dy);
        checkCuda(cudaDeviceSynchronize());



        // Calcular la presión
        solvePressurePoisson<<<numBlocks, threadsPerBlock>>>(p, divergence, nx, ny, dx, dy, 50);
        checkCuda(cudaDeviceSynchronize()); 



        // Aplicar el obstáculo
        applyObstacle<<<numBlocks, threadsPerBlock>>>(u_new, v_new, p, nx, ny, x1, y1, x2, y2);
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
    checkCuda(cudaFree(p));
    checkCuda(cudaFree(magnitude));

    return 0;
}
