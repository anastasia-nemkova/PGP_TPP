#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

#include "src/structures.cuh"
#include "src/camera.cuh"
#include "src/utils.cuh"
#include "src/figures.cuh"
#include "src/ray_tracing.cuh"
#include "src/ssaa.cuh"

using namespace std;
using namespace chrono;

Scene scene;
Camera camera;
int frames_to_render;
int width, height;
char output_path[256];
int ssaa_samples;

void parseInput() {
    cin >> frames_to_render;
    cin >> output_path;
    cin >> width >> height >> camera.fov;
    
    cin >> camera.r_c0 >> camera.z_c0 >> camera.phi_c0;
    cin >> camera.A_cr >> camera.A_cz;
    cin >> camera.w_cr >> camera.w_cz >> camera.w_cphi;
    cin >> camera.p_cr >> camera.p_cz;
    
    cin >> camera.r_n0 >> camera.z_n0 >> camera.phi_n0;
    cin >> camera.A_nr >> camera.A_nz;
    cin >> camera.w_nr >> camera.w_nz >> camera.w_nphi;
    cin >> camera.p_nr >> camera.p_nz;
    
    for (int i = 0; i < 3; i++) {
        Figure fig;
        cin >> fig.center.x >> fig.center.y >> fig.center.z;
        cin >> fig.color.x >> fig.color.y >> fig.color.z;
        cin >> fig.radius;
        cin >> fig.reflection_coef;
        cin >> fig.transparent_coef;
        cin >> fig.light_sources_number;
        fig.type = i;
        scene.figures.push_back(fig);
    }
    
    for (int i = 0; i < 4; i++) {
        cin >> scene.floor.points[i].x >> scene.floor.points[i].y >> scene.floor.points[i].z;
    }
    cin >> scene.floor.texture_path;
    cin >> scene.floor.color.x >> scene.floor.color.y >> scene.floor.color.z;
    cin >> scene.floor.reflection;
    
    int light_count;
    cin >> light_count;
    
    for (int i = 0; i < light_count; i++) {
        Light light;
        cin >> light.position.x >> light.position.y >> light.position.z;
        cin >> light.color.x >> light.color.y >> light.color.z;
        scene.lights.push_back(light);
    }
    
    cin >> scene.max_recursion_depth >> ssaa_samples;
}

void saveImage(const char* path, const Vec3* image, int width, int height, int frame) {
    char filename[256];
    sprintf(filename, path, frame);
    
    FILE* f = fopen(filename, "wb");
    if (!f) {
        cerr << "Cannot open file: " << filename << endl;
        return;
    }

    fwrite(&width, sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);

    for (int i = 0; i < width * height; i++) {
        Vec3 color = image[i];
        color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
        color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
        color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
        fwrite(&color.x, sizeof(float), 1, f);
        fwrite(&color.y, sizeof(float), 1, f);
        fwrite(&color.z, sizeof(float), 1, f);
    }
    
    fclose(f);
}

void defaultConfig() {
    cout << "1024" << endl;
    cout << "output/img_%d.data" << endl;
    cout << "640 480 120" << endl;
    
    cout << "7.0 3.0 0.0 2.0 1.0 2.0 1.0 0.0 0.0" << endl;
    cout << "2.0 0.0 0.0 0.5 0.1 1.0 1.0 0.0 0.0" << endl;
    
    cout << "2.0 0.0 0.0 1.0 0.0 0.0 1.0 0.9 0.1 10" << endl;
    cout << "0.0 2.0 0.0 0.0 1.0 0.0 0.75 0.8 0.2 5" << endl;
    cout << "0.0 0.0 0.0 0.0 0.7 0.7 0.5 0.7 0.3 2" << endl;
    
    cout << "-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0" << endl;
    cout << "texture/floor.data" << endl;
    cout << "0.0 1.0 0.0 0.5" << endl;
    
    cout << "2" << endl;
    cout << "-10.0 0.0 10.0 1.0 1.0 1.0" << endl;
    cout << "1.0 0.0 10.0 0.0 0.0 1.0" << endl;
    
    cout << "10 16" << endl;
}

int main(int argc, char** argv) {
    bool use_gpu = true;
    bool default_config = false;
    
    if (argc == 1) {
        use_gpu = true;
    } else {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--cpu") == 0) {
                use_gpu = false;
            }
            else if (strcmp(argv[i], "--gpu") == 0) {
                use_gpu = true;
            }
            else if (strcmp(argv[i], "--default") == 0) {
                default_config = true;
            }
        }
    }
    
    if (default_config) {
        defaultConfig();
        return 0;
    }
    
    parseInput();
    camera.update(0.0f); 

    if (!loadTexture(scene.floor.texture_path, scene.floor)) {
        cout << "Using solid color for floor" << endl;
    }
    
    buildScene(scene);  
    
    if (use_gpu) {
        scene.allocateGPU();
    }
    
    double total_time = 0;
    long long total_rays = 0;
    
    cout << "Frame\tTime(ms)\tTotal Rays" << endl;
    cout << string(40, '-') << endl;
    
    for (int frame = 0; frame < frames_to_render; frame++) {
        auto start = high_resolution_clock::now();
        
        float t = (float)frame / frames_to_render * 2.0f * M_PI;

        camera.update(t);
        
        Vec3* image = new Vec3[width * height];
        long long frame_rays = 0;
        
        try {
            if (use_gpu) {
                renderFrameGPU(image, scene, camera, width, height, ssaa_samples, frame_rays);
            } else {
                renderFrameCPU(image, scene, camera, width, height, ssaa_samples, frame_rays);
            }
            
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            double frame_time = duration.count() / 1000.0;
            total_time += frame_time;
            total_rays += frame_rays;
            
            saveImage(output_path, image, width, height, frame);
            
            cout << frame << "\t" 
                 << fixed << setprecision(1) << frame_time << "\t"
                 << frame_rays << endl;
            
            delete[] image;
            
        } catch (const exception& e) {
            cerr << "Error rendering frame " << frame << ": " << e.what() << endl;
            delete[] image;
            break;
        }
    }
    
    if (use_gpu) {
        scene.freeGPU();
    }
    
    cout << string(40, '-') << endl;
    cout << "Total time: " << fixed << setprecision(1) << total_time << " ms" << endl;
    if (total_time > 0) {
        cout << "Average FPS: " << fixed << setprecision(2) 
             << frames_to_render / (total_time / 1000.0) << endl;
    }
    cout << "Total rays: " << total_rays << endl;
    
    return 0;
}