#ifndef __RAY_TRACING_CUH__
#define __RAY_TRACING_CUH__

#include "structures.cuh"
#include "utils.cuh"
#include "figures.cuh"
#include "ssaa.cuh"
#include <vector>
#include <algorithm>

#define BLOCKS 128
#define THREADS 128
#define BLOCKS_2D dim3(32, 32)
#define THREADS_2D dim3(16, 16)
#define POLY_COEF 1e-4f
#define SHADOW_BIAS 1e-3f


bool cpuFindIntersection(const Ray& ray, const std::vector<Polygon>& polygons, Intersection& best_intersection) {
    best_intersection.reset();
    bool found = false;
    
    for (size_t i = 0; i < polygons.size(); i++) {
        float t, u, v;
        
        if (rayTriangleIntersect(ray, polygons[i].v0, polygons[i].v1, polygons[i].v2, t, u, v)) {
            if (t < best_intersection.t && t > EPSILON) {
                best_intersection.t = t;
                best_intersection.object_id = polygons[i].object_id;
                best_intersection.polygon_id = (int)i;
                best_intersection.point = ray.pointAt(t);
                best_intersection.normal = polygons[i].normal;
                best_intersection.color = polygons[i].color;
                best_intersection.reflection = polygons[i].reflection;
                best_intersection.transparency = polygons[i].transparency;
                best_intersection.is_floor = polygons[i].is_floor;
                
                if (best_intersection.normal.dot(ray.direction) > 0) {
                    best_intersection.normal = -best_intersection.normal;
                }
                
                found = true;
            }
        }
    }
    
    return found;
}

bool cpuCheckShadow(const Ray& shadow_ray, float max_distance, const std::vector<Polygon>& polygons) {
    
    for (size_t i = 0; i < polygons.size(); i++) {
        float t;
        if (rayTriangleIntersect(shadow_ray, polygons[i].v0, polygons[i].v1, polygons[i].v2, t)) {
            if (t > SHADOW_BIAS && t < max_distance - SHADOW_BIAS) {
                if (polygons[i].transparency < 0.8f) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

Vec3 cpuComputeShading(const Intersection& inter, const Vec3& view_dir, const std::vector<Light>& lights, const std::vector<Polygon>& polygons, const Floor& floor, const Vec3& hit_point) {
    Vec3 color(0, 0, 0);
    Vec3 normal = inter.normal.normalized();

    Vec3 surface_color = inter.color;

    if (inter.is_floor) {
        surface_color = floor.getColor(hit_point);
    }

    float ambient = 0.1f;
    color = color + surface_color * ambient;

    for (const auto& light : lights) {
        Vec3 light_dir = (light.position - hit_point);
        float distance = light_dir.length();
        light_dir = light_dir / distance;
        
        Ray shadow_ray(hit_point + normal * POLY_COEF, light_dir);
        bool in_shadow = cpuCheckShadow(shadow_ray, distance, polygons);
        
        if (!in_shadow) {

            float diff = fmaxf(0.0f, normal.dot(light_dir));
            Vec3 diffuse = surface_color * light.color * diff;

            if (inter.reflection > 0.0f) {
                Vec3 half_dir = (light_dir + view_dir).normalized();
                float spec = powf(fmaxf(0.0f, normal.dot(half_dir)), 32.0f);
                Vec3 specular = light.color * spec * inter.reflection;
                color = color + specular;
            }
            
            color = color + diffuse;
        }
    }

    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
    
    return color;
}

Vec3 cpuTraceRaySingleBounce(const Ray& ray, const std::vector<Polygon>& polygons, const std::vector<Light>& lights, const Floor& floor) {
    
    Intersection primary_inter;

    if (!cpuFindIntersection(ray, polygons, primary_inter)) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    if (primary_inter.object_id == -2) {
        return primary_inter.color * 15.0f;
    }
    
    Vec3 hit_point = primary_inter.point;
    Vec3 normal = primary_inter.normal.normalized();
    Vec3 view_dir = -ray.direction.normalized();
    

    Vec3 direct_color = cpuComputeShading(primary_inter, view_dir, lights, polygons, floor, hit_point);
    
    Vec3 final_color = direct_color;

    if (primary_inter.transparency > 0.0f) {
        Ray continue_ray(hit_point - normal * POLY_COEF * 2.0f, ray.direction);

        Intersection behind_inter;
        if (cpuFindIntersection(continue_ray, polygons, behind_inter)) {
            Vec3 behind_hit = behind_inter.point;
            Vec3 behind_view = -continue_ray.direction.normalized();
            Vec3 behind_color = cpuComputeShading(behind_inter, behind_view, lights, polygons, floor, behind_hit);

            final_color = direct_color * (1.0f - primary_inter.transparency) + behind_color * primary_inter.transparency;
        }
    } 
    else if (primary_inter.reflection > 0.0f) {
        Vec3 reflect_dir = reflect(ray.direction, normal).normalized();
        Ray reflect_ray(hit_point + normal * POLY_COEF, reflect_dir);

        Intersection reflect_inter;
        if (cpuFindIntersection(reflect_ray, polygons, reflect_inter)) {
            Vec3 reflect_hit = reflect_inter.point;
            Vec3 reflect_view = -reflect_ray.direction.normalized();
            Vec3 reflect_color = cpuComputeShading(reflect_inter, reflect_view, lights, polygons, floor, reflect_hit);

            final_color = direct_color * (1.0f - primary_inter.reflection) + reflect_color * primary_inter.reflection;
        }
    }
    
    return final_color;
}


void cpuRenderFrame(Vec3* output_image, const Scene& scene, const Camera& camera, int width, int height, int ssaa_sqrt, long long& total_rays, int frame_id) {

    if (ssaa_sqrt <= 1) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel_id = y * width + x;
                Ray ray = camera.getRay(x, y);
                output_image[pixel_id] = cpuTraceRaySingleBounce(ray, scene.polygons, 
                                                               scene.lights, scene.floor);
            }
        }
        total_rays = (long long)width * height;
    } else {
        int sample_count = ssaa_sqrt * ssaa_sqrt;
        float inv_sample_count = 1.0f / sample_count;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Vec3 pixel_color(0, 0, 0);
                
                for (int sy = 0; sy < ssaa_sqrt; sy++) {
                    for (int sx = 0; sx < ssaa_sqrt; sx++) {
                        float offset_x = (sx + 0.5f) / ssaa_sqrt - 0.5f;
                        float offset_y = (sy + 0.5f) / ssaa_sqrt - 0.5f;
                        
                        Ray ray = camera.getRay(x + offset_x, y + offset_y);
                        pixel_color = pixel_color + cpuTraceRaySingleBounce(ray, scene.polygons, scene.lights, scene.floor) * inv_sample_count;
                    }
                }
                
                output_image[y * width + x] = pixel_color;
            }
        }
        total_rays = (long long)width * height * sample_count;
    }
}


__device__ bool gpuFindIntersection(const Ray& ray, const Polygon* polygons, int polygons_number, Intersection& best_intersection) {
    
    best_intersection.reset();
    bool found = false;
    
    for (int i = 0; i < polygons_number; i++) {
        float t, u, v;
        
        if (rayTriangleIntersect(ray, polygons[i].v0, polygons[i].v1, polygons[i].v2, t, u, v)) {
            if (t < best_intersection.t && t > EPSILON) {
                best_intersection.t = t;
                best_intersection.object_id = polygons[i].object_id;
                best_intersection.polygon_id = i;
                best_intersection.point = ray.pointAt(t);
                best_intersection.normal = polygons[i].normal;
                best_intersection.color = polygons[i].color;
                best_intersection.reflection = polygons[i].reflection;
                best_intersection.transparency = polygons[i].transparency;
                best_intersection.is_floor = polygons[i].is_floor;
                
                if (best_intersection.normal.dot(ray.direction) > 0) {
                    best_intersection.normal = -best_intersection.normal;
                }
                
                found = true;
            }
        }
    }
    
    return found;
}

__device__ bool gpuCheckShadow(const Ray& shadow_ray,
                              float max_distance,
                              const Polygon* polygons, int polygons_number) {
    
    for (int i = 0; i < polygons_number; i++) {
        float t;
        if (rayTriangleIntersect(shadow_ray, polygons[i].v0, polygons[i].v1, polygons[i].v2, t)) {
            if (t > SHADOW_BIAS && t < max_distance - SHADOW_BIAS) {
                if (polygons[i].transparency < 0.8f) {
                    return true;
                }
            }
        }
    }
    
    return false;
}


__device__ Vec3 gpuGetTextureColor(const Vec3* tex_data, int tex_width, int tex_height, const Vec2& uv, const Vec3& default_color) {
    
    if (tex_data == nullptr || tex_width == 0 || tex_height == 0) {
        return default_color;
    }
    
    float u = uv.u - floorf(uv.u);
    float v = uv.v - floorf(uv.v);
    if (u < 0) u += 1.0f;
    if (v < 0) v += 1.0f;
    
    int x = (int)(u * tex_width);
    int y = (int)(v * tex_height);
    
    x = max(0, min(x, tex_width - 1));
    y = max(0, min(y, tex_height - 1));
    
    return tex_data[y * tex_width + x];
}

__device__ Vec3 gpuComputeShading(const Intersection& inter, const Vec3& view_dir, const Light* lights, int light_count, const Polygon* polygons, int polygons_number, const Floor& floor, const Vec3* floor_tex, int floor_tex_width, int floor_tex_height, const Vec3& hit_point) {
    
    Vec3 color(0, 0, 0);
    Vec3 normal = inter.normal.normalized();

    Vec3 surface_color = inter.color;

    if (inter.is_floor && floor_tex != nullptr && floor_tex_width > 0 && floor_tex_height > 0) {
        Vec2 uv = floor.getUV(hit_point);
        surface_color = gpuGetTextureColor(floor_tex, floor_tex_width, floor_tex_height, uv, floor.color);
    }

    float ambient = 0.1f;
    color = color + surface_color * ambient;

    for (int i = 0; i < light_count; i++) {
        Light light = lights[i];
        Vec3 light_dir = (light.position - hit_point);
        float distance = light_dir.length();
        light_dir = light_dir / distance;
        
        Ray shadow_ray(hit_point + normal * POLY_COEF, light_dir);
        bool in_shadow = gpuCheckShadow(shadow_ray, distance, polygons, polygons_number);
        
        if (!in_shadow) {
            float diff = fmaxf(0.0f, normal.dot(light_dir));
            Vec3 diffuse = surface_color * light.color * diff;
            
            if (inter.reflection > 0.0f) {
                Vec3 half_dir = (light_dir + view_dir).normalized();
                float spec = powf(fmaxf(0.0f, normal.dot(half_dir)), 32.0f);
                Vec3 specular = light.color * spec * inter.reflection;
                color = color + specular;
            }
            
            color = color + diffuse;
        }
    }

    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
    
    return color;
}

__device__ Vec3 gpuTraceRaySingleBounce(const Ray& ray, const Polygon* polygons, int polygons_number, const Light* lights, int light_count, const Floor& floor, const Vec3* floor_tex, int floor_tex_width, int floor_tex_height) {
    
    Intersection primary_inter;
    
    if (!gpuFindIntersection(ray, polygons, polygons_number, primary_inter)) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    if (primary_inter.object_id == -2) {
        return primary_inter.color * 15.0f;
    }
    
    Vec3 hit_point = primary_inter.point;
    Vec3 normal = primary_inter.normal.normalized();
    Vec3 view_dir = -ray.direction.normalized();

    Vec3 direct_color = gpuComputeShading(primary_inter, view_dir, lights, light_count, 
                                         polygons, polygons_number, floor,
                                         floor_tex, floor_tex_width, floor_tex_height, hit_point);
    
    Vec3 final_color = direct_color;

    if (primary_inter.transparency > 0.0f) {
        Ray continue_ray(hit_point - normal * POLY_COEF * 2.0f, ray.direction);
        
        Intersection behind_inter;
        if (gpuFindIntersection(continue_ray, polygons, polygons_number, behind_inter)) {
            Vec3 behind_hit = behind_inter.point;
            Vec3 behind_view = -continue_ray.direction.normalized();
            Vec3 behind_color = gpuComputeShading(behind_inter, behind_view, lights, light_count,
                                                 polygons, polygons_number, floor,
                                                 floor_tex, floor_tex_width, floor_tex_height, behind_hit);
            
            final_color = direct_color * (1.0f - primary_inter.transparency) + behind_color * primary_inter.transparency;
        }
    } else if (primary_inter.reflection > 0.0f) {
        Vec3 reflect_dir = reflect(ray.direction, normal).normalized();
        Ray reflect_ray(hit_point + normal * POLY_COEF, reflect_dir);
        
        Intersection reflect_inter;
        if (gpuFindIntersection(reflect_ray, polygons, polygons_number, reflect_inter)) {
            Vec3 reflect_hit = reflect_inter.point;
            Vec3 reflect_view = -reflect_ray.direction.normalized();
            Vec3 reflect_color = gpuComputeShading(reflect_inter, reflect_view, lights, light_count,
                                                  polygons, polygons_number, floor,
                                                  floor_tex, floor_tex_width, floor_tex_height, reflect_hit);
            
            final_color = direct_color * (1.0f - primary_inter.reflection) + reflect_color * primary_inter.reflection;
        }
    }
    
    return final_color;
}

__global__ void gpuTraceKernelSingleBounce(
    Vec3* image_data,
    const Polygon* polygons, int polygons_number,
    const Light* lights, int light_count,
    const Floor floor,
    const Vec3* floor_tex, int floor_tex_width, int floor_tex_height,
    const Camera camera,
    int width, int height,
    int ssaa_sqrt) {
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int total_pixels = width * height;
    
    for (int pixel_id = thread_id; pixel_id < total_pixels; pixel_id += total_threads) {
        int y = pixel_id / width;
        int x = pixel_id % width;
        
        if (ssaa_sqrt <= 1) {
            Ray ray = camera.getRay(x, y);
            image_data[pixel_id] = gpuTraceRaySingleBounce(ray, polygons, polygons_number, lights, light_count, floor, floor_tex, floor_tex_width, floor_tex_height);
        } else {
            int sample_count = ssaa_sqrt * ssaa_sqrt;
            float inv_sample_count = 1.0f / sample_count;
            Vec3 pixel_color(0, 0, 0);
            
            for (int sy = 0; sy < ssaa_sqrt; sy++) {
                for (int sx = 0; sx < ssaa_sqrt; sx++) {
                    float offset_x = (sx + 0.5f) / ssaa_sqrt - 0.5f;
                    float offset_y = (sy + 0.5f) / ssaa_sqrt - 0.5f;
                    
                    Ray ray = camera.getRay(x + offset_x, y + offset_y);
                    pixel_color = pixel_color + gpuTraceRaySingleBounce(ray, polygons, polygons_number,  lights, light_count, floor, floor_tex, floor_tex_width, floor_tex_height) * inv_sample_count;
                }
            }
            
            image_data[pixel_id] = pixel_color;
        }
    }
}
__global__ void gpuConvertToUChar4(Vec3* vec3_data, uchar4* uchar4_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    Vec3 color = vec3_data[idx];
    
    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

    color.x = powf(color.x, 1.0f/2.2f);
    color.y = powf(color.y, 1.0f/2.2f);
    color.z = powf(color.z, 1.0f/2.2f);
    
    uchar4_data[idx] = make_uchar4(
        (unsigned char)(color.x * 255.0f),
        (unsigned char)(color.y * 255.0f),
        (unsigned char)(color.z * 255.0f),
        255);
}

void gpuRenderFrame(Vec3* output_image,
                   const Scene& scene,
                   const Camera& camera,
                   int width, int height,
                   int ssaa_sqrt,
                   long long& total_rays,
                   int frame_id) {
    
    int hr_width = width * ssaa_sqrt;
    int hr_height = height * ssaa_sqrt;
    int hr_size = hr_width * hr_height;

    Camera hr_camera = camera;
    hr_camera.setResolution(hr_width, hr_height);
    hr_camera.setFOV(camera.getFOV());
    
    Vec3* d_image;
    CSC(cudaMalloc(&d_image, hr_size * sizeof(Vec3)));

    Floor d_floor = scene.floor;

    Vec3* d_floor_tex = nullptr;
    int floor_tex_width = 0, floor_tex_height = 0;
    
    if (scene.floor.has_texture && scene.floor.tex_data) {
        floor_tex_width = scene.floor.tex_width;
        floor_tex_height = scene.floor.tex_height;
        int tex_size = floor_tex_width * floor_tex_height * sizeof(Vec3);
        CSC(cudaMalloc(&d_floor_tex, tex_size));
        CSC(cudaMemcpy(d_floor_tex, scene.floor.tex_data, tex_size, cudaMemcpyHostToDevice));
    }
    
    gpuTraceKernelSingleBounce<<<BLOCKS, THREADS>>>(
        d_image,
        scene.d_polygons, scene.polygon_count,
        scene.d_lights, scene.light_count,
        d_floor,
        d_floor_tex, floor_tex_width, floor_tex_height,
        hr_camera,
        hr_width, hr_height,
        1
    );
    
    CSC(cudaDeviceSynchronize());
    
    if (ssaa_sqrt > 1) {
        uchar4* d_hr_uchar;
        CSC(cudaMalloc(&d_hr_uchar, hr_size * sizeof(uchar4)));
        gpuConvertToUChar4<<<(hr_size + THREADS - 1)/THREADS, THREADS>>>(d_image, d_hr_uchar, hr_size);
        
        uchar4* d_lr_uchar;
        CSC(cudaMalloc(&d_lr_uchar, width * height * sizeof(uchar4)));
        applySSAA_GPU(d_hr_uchar, d_lr_uchar, width, height, ssaa_sqrt);
        
        uchar4* h_lr_uchar = new uchar4[width * height];
        CSC(cudaMemcpy(h_lr_uchar, d_lr_uchar, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < width * height; i++) {
            output_image[i] = Vec3(
                h_lr_uchar[i].x / 255.0f,
                h_lr_uchar[i].y / 255.0f,
                h_lr_uchar[i].z / 255.0f
            );
        }
        
        CSC(cudaFree(d_hr_uchar));
        CSC(cudaFree(d_lr_uchar));
        delete[] h_lr_uchar;
    } else {
        CSC(cudaMemcpy(output_image, d_image, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost));
    }

    if (d_floor_tex) CSC(cudaFree(d_floor_tex));
    CSC(cudaFree(d_image));
    
    total_rays = (long long)hr_width * hr_height * ((ssaa_sqrt > 1) ? ssaa_sqrt * ssaa_sqrt : 1);
}

void renderFrameCPU(Vec3* output_image, const Scene& scene, const Camera& camera, 
                   int width, int height, int ssaa_sqrt, long long& total_rays) {
    cpuRenderFrame(output_image, scene, camera, width, height, ssaa_sqrt, total_rays, 0);
}

void renderFrameGPU(Vec3* output_image, const Scene& scene, const Camera& camera, 
                   int width, int height, int ssaa_sqrt, long long& total_rays) {
    gpuRenderFrame(output_image, scene, camera, width, height, ssaa_sqrt, total_rays, 0);
}

#endif // __RAY_TRACING_CUH__