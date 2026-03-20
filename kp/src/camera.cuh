#ifndef __CAMERA_CUH__
#define __CAMERA_CUH__

#include "structures.cuh"
#include <cmath>

class Camera {
public:
    float r_c0, z_c0, phi_c0;
    float A_cr, A_cz;
    float w_cr, w_cz, w_cphi;
    float p_cr, p_cz;

    float r_n0, z_n0, phi_n0;
    float A_nr, A_nz;
    float w_nr, w_nz, w_nphi;
    float p_nr, p_nz;
    
    Vec3 position;
    Vec3 target;
    Vec3 up;
    Vec3 forward, right;
    
    float fov;
    int width, height;
    float aspect_ratio;
    float tan_fov;

public:
    Camera() : fov(60.0f), width(640), height(480) {
        update(0.0f);
    }

    void setMovementParams(
        float rc0, float zc0, float phic0,
        float Acr, float Acz,
        float wcr, float wcz, float wcphi,
        float pcr, float pcz,
        float rn0, float zn0, float phin0,
        float Anr, float Anz,
        float wnr, float wnz, float wnphi,
        float pnr, float pnz) {
        
        r_c0 = rc0; z_c0 = zc0; phi_c0 = phic0;
        A_cr = Acr; A_cz = Acz;
        w_cr = wcr; w_cz = wcz; w_cphi = wcphi;
        p_cr = pcr; p_cz = pcz;
        
        r_n0 = rn0; z_n0 = zn0; phi_n0 = phin0;
        A_nr = Anr; A_nz = Anz;
        w_nr = wnr; w_nz = wnz; w_nphi = wnphi;
        p_nr = pnr; p_nz = pnz;
    }
    
    void setResolution(int w, int h) {
        width = w;
        height = h;
        aspect_ratio = (float)width / (float)height;
    }
    
    void setFOV(float fov_degrees) {
        fov = fov_degrees;
        float fov_rad = fov * PI / 180.0f;
        tan_fov = tanf(fov_rad / 2.0f);
    }
    
    void update(float t) {
        float r_c = r_c0 + A_cr * sinf(w_cr * t + p_cr);
        float z_c = z_c0 + A_cz * sinf(w_cz * t + p_cz);
        float phi_c = phi_c0 + w_cphi * t;

        float x_c = r_c * cosf(phi_c);
        float y_c = r_c * sinf(phi_c);
        
        position = Vec3(x_c, y_c, z_c);

        float r_n = r_n0 + A_nr * sinf(w_nr * t + p_nr);
        float z_n = z_n0 + A_nz * sinf(w_nz * t + p_nz);
        float phi_n = phi_n0 + w_nphi * t;
        
        float x_n = r_n * cosf(phi_n);
        float y_n = r_n * sinf(phi_n);
        
        target = Vec3(x_n, y_n, z_n);

        forward = (target - position).normalized();
        up = Vec3(0, 0, 1);
        right = forward.cross(up).normalized();
        up = right.cross(forward).normalized();

        aspect_ratio = (float)width / (float)height;
        float fov_rad = fov * PI / 180.0f;
        tan_fov = tanf(fov_rad / 2.0f);
    }
    
    __host__ __device__ Ray getRay(float x, float y) const {
        float u = (2.0f * (x + 0.5f) / width - 1.0f) * aspect_ratio * tan_fov;
        float v = (1.0f - 2.0f * (y + 0.5f) / height) * tan_fov;
        
        Vec3 direction = (forward + right * u + up * v).normalized();
        return Ray(position, direction);
    }
    
    const Vec3& getPosition() const { return position; }
    const Vec3& getTarget() const { return target; }
    const Vec3& getForward() const { return forward; }
    const Vec3& getRight() const { return right; }
    const Vec3& getUp() const { return up; }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    float getFOV() const { return fov; }
};

#endif // __CAMERA_CUH__