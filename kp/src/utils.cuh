#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "structures.cuh"
#include "camera.cuh"

__host__ bool loadTexture(const char* filename, Floor& floor) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Warning: Cannot open texture file: %s\n", filename);
        printf("Using solid color for floor\n");
        floor.has_texture = false;
        return false;
    }

    int width, height;
    if (fread(&width, sizeof(int), 1, f) != 1 ||
        fread(&height, sizeof(int), 1, f) != 1) {
        printf("Warning: Cannot read texture dimensions\n");
        fclose(f);
        floor.has_texture = false;
        return false;
    }
    
    unsigned char* rgba_data = new unsigned char[width * height * 4];
    size_t read_count = fread(rgba_data, 1, width * height * 4, f);
    fclose(f);
    
    if (read_count != width * height * 4) {
        printf("Warning: Cannot read RGBA texture data\n");
        delete[] rgba_data;
        floor.has_texture = false;
        return false;
    }

    floor.tex_width = width;
    floor.tex_height = height;
    floor.tex_data = new Vec3[width * height];
    
    for (int i = 0; i < width * height; i++) {
        float r = rgba_data[i * 4] / 255.0f;
        float g = rgba_data[i * 4 + 1] / 255.0f;
        float b = rgba_data[i * 4 + 2] / 255.0f;
        floor.tex_data[i] = Vec3(r, g, b);
    }
    
    delete[] rgba_data;
    floor.has_texture = true;
    floor.computeBoundingBox();
    return true;
}

__host__ void freeTexture(Floor& floor) {
    if (floor.tex_data && floor.has_texture) {
        delete[] floor.tex_data;
        floor.tex_data = nullptr;
        floor.has_texture = false;
    }
}

__host__ __device__ inline float clamp(float x, float min_val = 0.0f, float max_val = 1.0f) {
    return fminf(fmaxf(x, min_val), max_val);
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * (2.0f * v.dot(n));
}

__host__ __device__ inline float deg2rad(float deg) {
    return deg * PI / 180.0f;
}

__host__ __device__ inline bool rayTriangleIntersect(
    const Ray& ray,
    const Vec3& v0, const Vec3& v1, const Vec3& v2,
    float& t, float& u, float& v) {
    
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = ray.direction.cross(edge2);
    float a = edge1.dot(h);
    
    if (a > -EPSILON && a < EPSILON)
        return false;
    
    float f = 1.0f / a;
    Vec3 s = ray.origin - v0;
    u = f * s.dot(h);
    
    if (u < 0.0f || u > 1.0f)
        return false;
    
    Vec3 q = s.cross(edge1);
    v = f * ray.direction.dot(q);
    
    if (v < 0.0f || u + v > 1.0f)
        return false;
    
    t = f * edge2.dot(q);
    return t > EPSILON;
}

__host__ __device__ inline bool rayTriangleIntersect(
    const Ray& ray,
    const Vec3& v0, const Vec3& v1, const Vec3& v2,
    float& t) {
    float u, v;
    return rayTriangleIntersect(ray, v0, v1, v2, t, u, v);
}

__host__ bool readConfig(const char* filename, Scene& scene, Camera& camera, 
                        int& width, int& height, int& num_frames, 
                        char* output_pattern, int& ssaa_sqrt) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open config file: %s\n", filename);
        return false;
    }

    if (fscanf(f, "%d", &num_frames) != 1) {
        printf("Error reading number of frames\n");
        fclose(f);
        return false;
    }
    
    if (fscanf(f, "%s", output_pattern) != 1) {
        printf("Error reading output pattern\n");
        fclose(f);
        return false;
    }
    
    if (fscanf(f, "%d %d", &width, &height) != 2) {
        printf("Error reading image dimensions\n");
        fclose(f);
        return false;
    }
    
    float fov;
    if (fscanf(f, "%f", &fov) != 1) {
        printf("Error reading FOV\n");
        fclose(f);
        return false;
    }
    camera.fov = fov;

    if (fscanf(f, "%f %f %f", &camera.r_c0, &camera.z_c0, &camera.phi_c0) != 3 ||
        fscanf(f, "%f %f", &camera.A_cr, &camera.A_cz) != 2 ||
        fscanf(f, "%f %f %f", &camera.w_cr, &camera.w_cz, &camera.w_cphi) != 3 ||
        fscanf(f, "%f %f", &camera.p_cr, &camera.p_cz) != 2 ||
        fscanf(f, "%f %f %f", &camera.r_n0, &camera.z_n0, &camera.phi_n0) != 3 ||
        fscanf(f, "%f %f", &camera.A_nr, &camera.A_nz) != 2 ||
        fscanf(f, "%f %f %f", &camera.w_nr, &camera.w_nz, &camera.w_nphi) != 3 ||
        fscanf(f, "%f %f", &camera.p_nr, &camera.p_nz) != 2) {
        printf("Error reading camera parameters\n");
        fclose(f);
        return false;
    }

    int num_figures = 3;
    scene.figures.resize(num_figures);
    
    for (int i = 0; i < num_figures; i++) {
        Figure& fig = scene.figures[i];
        if (fscanf(f, "%f %f %f", &fig.center.x, &fig.center.y, &fig.center.z) != 3 ||
            fscanf(f, "%f %f %f", &fig.color.x, &fig.color.y, &fig.color.z) != 3 ||
            fscanf(f, "%f %f %f %d", &fig.radius, &fig.reflection_coef, 
                   &fig.transparent_coef, &fig.light_sources_number) != 4) {
            printf("Error reading figure %d parameters\n", i);
            fclose(f);
            return false;
        }
        fig.type = i; 
    }

    for (int i = 0; i < 4; i++) {
        if (fscanf(f, "%f %f %f", &scene.floor.points[i].x, 
                  &scene.floor.points[i].y, &scene.floor.points[i].z) != 3) {
            printf("Error reading floor point %d\n", i);
            fclose(f);
            return false;
        }
    }
    
    char texture_path[256];
    if (fscanf(f, "%s", texture_path) != 1) {
        printf("Error reading texture path\n");
        fclose(f);
        return false;
    }
    strcpy(scene.floor.texture_path, texture_path);
    
    if (fscanf(f, "%f %f %f %f", &scene.floor.color.x, &scene.floor.color.y, 
              &scene.floor.color.z, &scene.floor.reflection) != 4) {
        printf("Error reading floor color/reflection\n");
        fclose(f);
        return false;
    }

    if (!loadTexture(texture_path, scene.floor)) {
        printf("Warning: Could not load texture, using solid color\n");
    }

    int num_lights;
    if (fscanf(f, "%d", &num_lights) != 1) {
        printf("Error reading number of lights\n");
        fclose(f);
        return false;
    }
    
    num_lights = min(num_lights, 4); 
    scene.lights.resize(num_lights);
    for (int i = 0; i < num_lights; i++) {
        Light& light = scene.lights[i];
        if (fscanf(f, "%f %f %f", &light.position.x, &light.position.y, &light.position.z) != 3 ||
            fscanf(f, "%f %f %f", &light.color.x, &light.color.y, &light.color.z) != 3) {
            printf("Error reading light %d parameters\n", i);
            fclose(f);
            return false;
        }
    }

    if (fscanf(f, "%d %d", &scene.max_recursion_depth, &ssaa_sqrt) != 2) {
        printf("Error reading recursion depth and SSAA\n");
        fclose(f);
        return false;
    }
    
    fclose(f);
    camera.width = width;
    camera.height = height;
    return true;
}

#endif