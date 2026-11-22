// Fused 3-D Yee updates for independent, static isotropic scenes and array shots.
// The same cell recurrence runs on the CPU and in CUDA. Arrays are component-major.
#include <algorithm>
#include <cstddef>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CELL __device__ __forceinline__
#else
#include <omp.h>
#define CELL inline
#endif

struct Layout {
    int nx, ny, nz, scenes, beams, antennas, path_length, samples, stride;
    double courant, direction[3];
};

struct Buffers {
    void *electric, *magnetic, *electric_history, *magnetic_history;
    const void *decay, *drive, *profiles, *source, *weights;
    const int *emitters, *paths;
    void *output;
};

template<typename T, bool electric, int axis>
CELL void derivative(const T* field, T* history, const T* profiles, int shot, int cell,
                     int coord, int extent, int offset, int volume, int profile_width,
                     T& first, T& second) {
    constexpr int tangent1 = (axis + 1) % 3;
    constexpr int tangent2 = (axis + 2) % 3;
    const T* f1 = field + std::size_t(tangent1) * volume;
    const T* f2 = field + std::size_t(tangent2) * volume;
    first = second = 0;
    if (electric ? coord > 0 : coord < extent - 1) {
        const int low = electric ? cell - offset : cell;
        const int high = electric ? cell : cell + offset;
        first = f1[high] - f1[low];
        second = f2[high] - f2[low];
    }
    const int base = ((electric ? 0 : 3) + axis) * 2 * profile_width + coord;
    const T c = profiles[base + profile_width];
    if (c != T(0)) {
        const T b = profiles[base];
        T* h1 = history + (std::size_t(shot) * 6 + 2 * axis) * volume + cell;
        T* h2 = h1 + volume;
        *h1 = b * *h1 + c * first;
        *h2 = b * *h2 + c * second;
        first += *h1;
        second += *h2;
    }
}

template<typename T, bool electric>
CELL void update_cell(const Layout& g, const Buffers& a, int shot, int cell, int step) {
    const int volume = g.nx * g.ny * g.nz;
    const int x = cell / (g.ny * g.nz);
    const int yz = cell - x * (g.ny * g.nz);
    const int y = yz / g.nz, z = yz - y * g.nz;
    const int max_xy = g.nx > g.ny ? g.nx : g.ny;
    const int width = max_xy > g.nz ? max_xy : g.nz;
    const std::size_t field_base = std::size_t(shot) * 3 * volume;
    const T* source_field = static_cast<const T*>(electric ? a.magnetic : a.electric) + field_base;
    T* out = static_cast<T*>(electric ? a.electric : a.magnetic) + field_base + cell;
    T* history = static_cast<T*>(electric ? a.electric_history : a.magnetic_history);
    const T* profiles = static_cast<const T*>(a.profiles);
    T xy, xz, yz_diff, yx, zx, zy;
    derivative<T,electric,0>(source_field, history, profiles, shot, cell, x, g.nx,
                             g.ny * g.nz, volume, width, xy, xz);
    derivative<T,electric,1>(source_field, history, profiles, shot, cell, y, g.ny,
                             g.nz, volume, width, yz_diff, yx);
    derivative<T,electric,2>(source_field, history, profiles, shot, cell, z, g.nz,
                             1, volume, width, zx, zy);
    const T curl[3] = {yz_diff - zy, zx - xz, xy - yx};
    if constexpr (electric) {
        const std::size_t material = std::size_t(shot / g.beams) * volume + cell;
        const T decay = static_cast<const T*>(a.decay)[material];
        const T drive = static_cast<const T*>(a.drive)[material];
        for (int k = 0; k < 3; ++k)
            out[k * volume] = decay * out[k * volume] + drive * curl[k];
        const int antenna = a.emitters[cell];
        if (antenna >= 0) {
            const std::size_t index = (std::size_t(step) * g.beams + shot % g.beams) * g.antennas + antenna;
            const T pulse = static_cast<const T*>(a.source)[index];
            for (int k = 0; k < 3; ++k)
                out[k * volume] += pulse * T(g.direction[k]);
        }
    } else {
        for (int k = 0; k < 3; ++k) out[k * volume] -= T(g.courant) * curl[k];
    }
}

template<typename T>
CELL void receive(const Layout& g, const Buffers& a, int shot, int antenna, int sample) {
    const int volume = g.nx * g.ny * g.nz;
    const T* field = static_cast<const T*>(a.electric) + std::size_t(shot) * 3 * volume;
    T voltage = 0;
    for (int j = 0; j < g.path_length; ++j) {
        const int path = antenna * g.path_length + j;
        const int cell = a.paths[path];
        T projected = 0;
        for (int k = 0; k < 3; ++k) projected += field[k * volume + cell] * T(g.direction[k]);
        voltage += projected * static_cast<const T*>(a.weights)[path];
    }
    const std::size_t trace = std::size_t(shot) * g.antennas + antenna;
    static_cast<T*>(a.output)[trace * g.samples + sample] = voltage;
}

#ifdef __CUDACC__
template<typename T, bool electric>
__global__ void half_step(Layout g, Buffers a, int step) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < g.nx * g.ny * g.nz) update_cell<T,electric>(g, a, blockIdx.y, cell, step);
}

template<typename T>
__global__ void gather(Layout g, Buffers a, int sample) {
    const int trace = blockIdx.x * blockDim.x + threadIdx.x;
    if (trace < g.scenes * g.beams * g.antennas)
        receive<T>(g, a, trace / g.antennas, trace % g.antennas, sample);
}
#endif

template<typename T>
int simulate(const Layout& g, const Buffers& a, void* stream_ptr) {
    const int shots = g.scenes * g.beams, volume = g.nx * g.ny * g.nz;
    const int steps = 1 + (g.samples - 1) * g.stride;
#ifdef __CUDACC__
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const dim3 grid((volume + 255) / 256, shots);
    for (int step = 0; step < steps; ++step) {
        half_step<T,true><<<grid,256,0,stream>>>(g, a, step);
        half_step<T,false><<<grid,256,0,stream>>>(g, a, step);
        if (step % g.stride == 0)
            gather<T><<<(shots * g.antennas + 127) / 128,128,0,stream>>>(g, a, step / g.stride);
    }
    return static_cast<int>(cudaGetLastError());
#else
    const int threads = std::min(shots, omp_get_max_threads());
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int shot = 0; shot < shots; ++shot) {
        for (int step = 0; step < steps; ++step) {
            for (int cell = 0; cell < volume; ++cell) update_cell<T,true>(g, a, shot, cell, step);
            for (int cell = 0; cell < volume; ++cell) update_cell<T,false>(g, a, shot, cell, step);
            if (step % g.stride == 0)
                for (int ant = 0; ant < g.antennas; ++ant) receive<T>(g, a, shot, ant, step / g.stride);
        }
    }
    return 0;
#endif
}

extern "C" int rf_simulate(const Layout* layout, const Buffers* buffers, int bytes, void* stream) {
    return bytes == 4 ? simulate<float>(*layout, *buffers, stream)
                      : simulate<double>(*layout, *buffers, stream);
}
