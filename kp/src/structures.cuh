#ifndef __STRUCTURES_CUH__
#define __STRUCTURES_CUH__

#include <vector>
#include <cmath>
#include <cstring>

#define PI 3.14159265358979323846f
#define EPSILON 1e-6f
#define INF 1e30f

#define CSC(call)                                                  \
do {                                                               \
    cudaError_t res = call;                                        \
    if (res != cudaSuccess) {                                      \
        printf("ERROR in %s:%d. Message: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(res));      \
        exit(0);                                                   \
    }                                                              \
} while(0)

struct Vec3 {
    float x, y, z;
    
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    __host__ __device__ Vec3 operator+(const Vec3& v) const { 
        return Vec3(x+v.x, y+v.y, z+v.z);
    }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { 
        return Vec3(x-v.x, y-v.y, z-v.z); 
    }
    __host__ __device__ Vec3 operator*(float s) const { 
        return Vec3(x*s, y*s, z*s); 
    }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { 
        return Vec3(x*v.x, y*v.y, z*v.z); 
    }
    __host__ __device__ Vec3 operator/(float s) const { 
        return Vec3(x/s, y/s, z/s); 
    }
    __host__ __device__ Vec3 operator-() const { 
        return Vec3(-x, -y, -z); 
    }
    
    __host__ __device__ Vec3& operator+=(const Vec3& v) { 
        x += v.x; y += v.y; z += v.z; 
        return *this; 
    }
    __host__ __device__ Vec3& operator-=(const Vec3& v) { 
        x -= v.x; y -= v.y; z -= v.z; 
        return *this; 
    }
    __host__ __device__ Vec3& operator*=(float s) { 
        x *= s; y *= s; z *= s; 
        return *this; 
    }
    __host__ __device__ Vec3& operator/=(float s) { 
        x /= s; y /= s; z /= s; 
        return *this; 
    }
    
    __host__ __device__ float length() const { 
        return sqrtf(x*x + y*y + z*z); 
    }
    __host__ __device__ float lengthSquared() const {
        return x*x + y*y + z*z;
    }
    __host__ __device__ Vec3 normalized() const { 
        float l = length(); 
        if (l > EPSILON) return *this / l; 
        return *this; 
    }
    
    __host__ __device__ float dot(const Vec3& v) const { 
        return x*v.x + y*v.y + z*v.z; 
    }
    __host__ __device__ Vec3 cross(const Vec3& v) const { 
        return Vec3(y*v.z - z*v.y, 
                   z*v.x - x*v.z, 
                   x*v.y - y*v.x); 
    }
};

__host__ __device__ inline Vec3 operator*(float s, const Vec3& v) {
    return Vec3(s*v.x, s*v.y, s*v.z);
}

struct Vec2 {
    float u, v;
    __host__ __device__ Vec2() : u(0), v(0) {}
    __host__ __device__ Vec2(float u, float v) : u(u), v(v) {}
    
    __host__ __device__ Vec2 operator+(const Vec2& other) const {
        return Vec2(u + other.u, v + other.v);
    }
    
    __host__ __device__ Vec2 operator*(float s) const {
        return Vec2(u * s, v * s);
    }
};

struct Vec3i {
    int x, y, z;
    __host__ __device__ Vec3i() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3i(int x, int y, int z) : x(x), y(y), z(z) {}
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalized()) {}
    
    __host__ __device__ Vec3 pointAt(float t) const {
        return origin + direction * t;
    }
};

struct Intersection {
    float t;
    int object_id;
    int polygon_id;
    Vec3 point;
    Vec3 normal;
    Vec3 color;
    float reflection;
    float transparency;
    bool is_floor;
    
    __host__ __device__ Intersection() : t(INF), object_id(-1), polygon_id(-1), 
                                        reflection(0), transparency(0),
                                        is_floor(false) {}
    
    __host__ __device__ bool hit() const { return polygon_id != -1 && t > EPSILON; }
    
    __host__ __device__ void reset() {
        t = INF;
        object_id = -1;
        polygon_id = -1;
    }
};

struct Polygon {
    Vec3 v0, v1, v2;
    Vec3 normal;
    Vec3 color;
    float reflection;
    float transparency;
    int object_id;
    bool is_floor;
    
    __host__ __device__ Polygon() : object_id(-1), reflection(0), transparency(0),
                                   is_floor(false) {}
    
    __host__ __device__ void computeNormal() {
        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        normal = e1.cross(e2).normalized();
    }
};

struct Figure {
    Vec3 center;
    Vec3 color;
    float radius;
    float reflection_coef;
    float transparent_coef;
    int light_sources_number;
    int type;
    
    __host__ __device__ Figure() : radius(1.0f), reflection_coef(0.0f), 
                                  transparent_coef(0.0f), light_sources_number(0),
                                  type(0) {}
};

struct Floor {
    Vec3 points[4];
    char texture_path[256];
    Vec3 color;
    float reflection;
    
    int tex_width, tex_height;
    Vec3* tex_data;
    Vec3* d_tex_data;
    bool has_texture;

    Vec3 normal;
    Vec3 u_axis, v_axis;
    Vec3 origin;
    float min_x, max_x, min_z, max_z;
    
    __host__ void computeBoundingBox() {
        min_x = points[0].x;
        max_x = points[0].x;
        min_z = points[0].z;
        max_z = points[0].z;
        
        for (int i = 1; i < 4; i++) {
            if (points[i].x < min_x) min_x = points[i].x;
            if (points[i].x > max_x) max_x = points[i].x;
            if (points[i].z < min_z) min_z = points[i].z;
            if (points[i].z > max_z) max_z = points[i].z;
        }
        Vec3 edge1 = points[1] - points[0];
        Vec3 edge2 = points[3] - points[0];
        normal = edge1.cross(edge2).normalized();

        if (normal.y < 0) normal = -normal;
 
        u_axis = (points[1] - points[0]).normalized();
        v_axis = (points[3] - points[0]).normalized();
        origin = points[0];
    }
    
    __host__ __device__ Vec2 getUV(const Vec3& point) const {
        Vec3 rel = point - origin;
        float u = rel.dot(u_axis) / (points[1] - points[0]).length();
        float v = rel.dot(v_axis) / (points[3] - points[0]).length();
        return Vec2(u, v);
    }
    
    __host__ __device__ Vec3 getTextureColor(const Vec2& uv, const Vec3* tex_data, int tex_width, int tex_height) const {
        if (!tex_data || tex_width == 0 || tex_height == 0) {
            return color;
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
    
    __host__ __device__ Vec3 getColor(const Vec3& point, const Vec3* tex_data = nullptr, int tex_width = 0, int tex_height = 0) const {
        if (!has_texture || !tex_data) {
            return color;
        }
        Vec2 uv = getUV(point);
        return getTextureColor(uv, tex_data, tex_width, tex_height);
    }
};

struct Light {
    Vec3 position;
    Vec3 color;
    
    __host__ __device__ Light() {}
    __host__ __device__ Light(const Vec3& p, const Vec3& c) 
        : position(p), color(c) {}
};

struct Scene {
    std::vector<Figure> figures;
    std::vector<Polygon> polygons;
    Floor floor;
    std::vector<Light> lights;
    
    int ssaa_sqrt;
    int max_recursion_depth;
    
    Polygon* d_polygons;
    Light* d_lights;
    int polygon_count;
    int light_count;
    
    __host__ void allocateGPU() {
        polygon_count = polygons.size();
        light_count = lights.size();
        
        CSC(cudaMalloc(&d_polygons, polygon_count * sizeof(Polygon)));
        CSC(cudaMalloc(&d_lights, light_count * sizeof(Light)));
        
        if (polygon_count > 0) {
            CSC(cudaMemcpy(d_polygons, polygons.data(), polygon_count * sizeof(Polygon), cudaMemcpyHostToDevice));
        }
        if (light_count > 0) {
            CSC(cudaMemcpy(d_lights, lights.data(), light_count * sizeof(Light), cudaMemcpyHostToDevice));
        }
        if (floor.has_texture && floor.tex_data && floor.tex_width > 0 && floor.tex_height > 0) {
            int tex_size = floor.tex_width * floor.tex_height * sizeof(Vec3);
            CSC(cudaMalloc(&floor.d_tex_data, tex_size));
            CSC(cudaMemcpy(floor.d_tex_data, floor.tex_data, tex_size, cudaMemcpyHostToDevice));
        }
    }
    
    __host__ void freeGPU() {
        if (d_polygons) cudaFree(d_polygons);
        if (d_lights) cudaFree(d_lights);
        if (floor.d_tex_data) cudaFree(floor.d_tex_data);
        
        d_polygons = nullptr;
        d_lights = nullptr;
        floor.d_tex_data = nullptr;
    }
};

#endif // __STRUCTURES_CUH__