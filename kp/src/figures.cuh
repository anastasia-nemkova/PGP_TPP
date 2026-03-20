#ifndef __FIGURES_CUH__
#define __FIGURES_CUH__

#include "structures.cuh"
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>

#define EDGE_THICKNESS 0.03f
#define EDGE_COLOR_VALUE 0.1f
#define LIGHT_INTENSITY 0.05f

const Vec3 EDGE_COLOR(EDGE_COLOR_VALUE, EDGE_COLOR_VALUE, EDGE_COLOR_VALUE);
const Vec3 WHITE_LIGHT(1.0f, 1.0f, 1.0f);
const float PHI = (1.0f + sqrtf(5.0f)) / 2.0f;


struct Face {
    std::vector<int> indices;
};

__host__ std::vector<Vec3> createTetrahedronVertices() {
    std::vector<Vec3> vertices;
    vertices.push_back(Vec3(1.0f, 0.0f, -1.0f/sqrtf(2.0f)));
    vertices.push_back(Vec3(-1.0f, 0.0f, -1.0f/sqrtf(2.0f)));
    vertices.push_back(Vec3(0.0f, 1.0f, 1.0f/sqrtf(2.0f)));
    vertices.push_back(Vec3(0.0f, -1.0f, 1.0f/sqrtf(2.0f)));
    return vertices;
}

__host__ std::vector<Face> createTetrahedronFaces() {
    std::vector<Face> faces;
    
    Face face1; face1.indices = {0, 2, 1}; faces.push_back(face1);
    Face face2; face2.indices = {0, 1, 3}; faces.push_back(face2);
    Face face3; face3.indices = {0, 3, 2}; faces.push_back(face3);
    Face face4; face4.indices = {1, 2, 3}; faces.push_back(face4);
    
    return faces;
}

__host__ std::vector<Vec3> createDodecahedronVertices() {
    std::vector<Vec3> vertices = {
        Vec3(-1.0f, 1.0f, 1.0f),
        Vec3(0.0f, (PHI - 1.0f), PHI),
        Vec3(0.0f, -(PHI - 1.0f), PHI), 
        Vec3(-1.0f, -1.0f, 1.0f),
        Vec3(-PHI, 0.0f, (PHI - 1)), 
        Vec3(1.0f, 1.0f, 1.0f),
        Vec3(PHI, 0.0f, (PHI - 1)), 
        Vec3(1.0f, -1.0f, 1.0f),
        Vec3((PHI - 1), PHI, 0.0f), 
        Vec3(-(PHI - 1), PHI, 0.0f),
        Vec3(-1.0f, 1.0f, -1.0f),
        Vec3(-PHI, 0.0f, -(PHI - 1)),
        Vec3(-1.0f, -1.0f, -1.0f),
        Vec3(-(PHI - 1), -PHI, 0.0f),
        Vec3((PHI - 1), -PHI, 0.0f),
        Vec3(1.0f, 1.0f, -1.0f),
        Vec3(0.0f, (PHI - 1), -PHI),
        Vec3(PHI, 0.0f, -(PHI - 1)),
        Vec3(0.0f, -(PHI - 1), -PHI),
        Vec3(1.0f, -1.0f, -1.0f)
    };
    
    return vertices;
}

__host__ std::vector<Face> createDodecahedronFaces() {
    std::vector<Face> faces;    
    int pentagon_faces[12][5] = {
        {0, 1, 2, 3, 4},
        {3, 2, 7, 14, 13}, 
        {4, 3, 13, 12, 11}, 
        {0, 4, 11, 10, 9}, 
        {10, 11, 12, 18, 16}, 
        {12, 13, 14, 19, 18}, 
        {1, 2, 7, 6, 5}, 
        {7, 14, 19, 17, 6},
        {18, 19, 17, 15, 16},
        {9, 10, 16, 15, 8}, 
        {9, 8, 5, 1, 0}, 
        {5, 6, 17, 15, 8}
    };

    for (int i = 0; i < 12; i++) {
        Face face;
        for (int j = 0; j < 5; j++) {
            face.indices.push_back(pentagon_faces[i][j]);
        }
        faces.push_back(face);
    }    
    return faces;
}

__host__ std::vector<Vec3> createIcosahedronVertices() {
    std::vector<Vec3> vertices;
    
    vertices.push_back(Vec3(0.0f, -0.525731f, 0.850651f));
    vertices.push_back(Vec3(0.850651f, 0.0f, 0.525731f));
    vertices.push_back(Vec3(0.850651f, 0.0f, -0.525731f));
    vertices.push_back(Vec3(-0.850651f, 0.0f, -0.525731f));
    vertices.push_back(Vec3(-0.850651f, 0.0f, 0.525731f));
    vertices.push_back(Vec3(-0.525731f, 0.850651f, 0.0f));
    vertices.push_back(Vec3(0.525731f, 0.850651f, 0.0f));
    vertices.push_back(Vec3(0.525731f, -0.850651f, 0.0f));
    vertices.push_back(Vec3(-0.525731f, -0.850651f, 0.0f));
    vertices.push_back(Vec3(0.0f, -0.525731f, -0.850651f));
    vertices.push_back(Vec3(0.0f, 0.525731f, -0.850651f));
    vertices.push_back(Vec3(0.0f, 0.525731f, 0.850651f));
    
    return vertices;
}

__host__ std::vector<Face> createIcosahedronFaces() {
    std::vector<Face> faces;
    
    int faces_indices[20][3] = {
        {1, 2, 6}, {1, 7, 2}, {3, 4, 5}, {4, 3, 8},
        {6, 5, 11}, {5, 6, 10}, {9, 10, 2}, {10, 9, 3},
        {7, 8, 9}, {8, 7, 0}, {11, 0, 1}, {0, 11, 4},
        {6, 2, 10}, {1, 6, 11}, {3, 5, 10}, {5, 4, 11},
        {2, 7, 9}, {7, 1, 0}, {3, 9, 8}, {4, 8, 0}
    };
    
    for (int i = 0; i < 20; i++) {
        Face face;
        face.indices.push_back(faces_indices[i][0]);
        face.indices.push_back(faces_indices[i][1]);
        face.indices.push_back(faces_indices[i][2]);
        faces.push_back(face);
    }
    
    return faces;
}



__host__ void createFace(const std::vector<Vec3>& vertices, const Face& face, const Vec3& color, int object_id, std::vector<Polygon>& polygons, float reflection, float transparency) {

    if (face.indices.size() < 3) return;

    if (face.indices.size() == 3) {
        Polygon glass_poly;
        int idx0 = face.indices[0];
        int idx1 = face.indices[1];
        int idx2 = face.indices[2];
        
        if (idx0 < 0 || idx0 >= (int)vertices.size() ||
            idx1 < 0 || idx1 >= (int)vertices.size() ||
            idx2 < 0 || idx2 >= (int)vertices.size()) {
            return;
        }
        
        glass_poly.v0 = vertices[idx0];
        glass_poly.v1 = vertices[idx1];
        glass_poly.v2 = vertices[idx2];
        glass_poly.computeNormal();
        
        glass_poly.color = color;
        glass_poly.object_id = object_id;
        glass_poly.reflection = reflection;
        glass_poly.transparency = transparency;
        glass_poly.is_floor = false;
        
        polygons.push_back(glass_poly);
    }
    else if (face.indices.size() == 5) {
        int idx0 = face.indices[0];
        int idx1 = face.indices[1];
        int idx2 = face.indices[2];
        int idx3 = face.indices[3];
        int idx4 = face.indices[4];

        Polygon glass_poly1;
        glass_poly1.v0 = vertices[idx0];
        glass_poly1.v1 = vertices[idx1];
        glass_poly1.v2 = vertices[idx2];
        glass_poly1.computeNormal();
        glass_poly1.color = color;
        glass_poly1.object_id = object_id;
        glass_poly1.reflection = reflection;
        glass_poly1.transparency = transparency;
        glass_poly1.is_floor = false;
        polygons.push_back(glass_poly1);

        Polygon glass_poly2;
        glass_poly2.v0 = vertices[idx0];
        glass_poly2.v1 = vertices[idx2];
        glass_poly2.v2 = vertices[idx3];
        glass_poly2.computeNormal();
        glass_poly2.color = color;
        glass_poly2.object_id = object_id;
        glass_poly2.reflection = reflection;
        glass_poly2.transparency = transparency;
        glass_poly2.is_floor = false;
        polygons.push_back(glass_poly2);

        Polygon glass_poly3;
        glass_poly3.v0 = vertices[idx0];
        glass_poly3.v1 = vertices[idx3];
        glass_poly3.v2 = vertices[idx4];
        glass_poly3.computeNormal();
        glass_poly3.color = color;
        glass_poly3.object_id = object_id;
        glass_poly3.reflection = reflection;
        glass_poly3.transparency = transparency;
        glass_poly3.is_floor = false;
        polygons.push_back(glass_poly3);
    }
}

__host__ void createEdge(const Vec3& v0, const Vec3& v1, int object_id, std::vector<Polygon>& polygons) {
    Vec3 edge_dir = (v1 - v0).normalized();
    Vec3 perp1, perp2;
    
    if (fabsf(edge_dir.x) > 0.5f) {
        perp1 = Vec3(-edge_dir.z, 0, edge_dir.x).normalized();
    } else {
        perp1 = Vec3(0, edge_dir.z, -edge_dir.y).normalized();
    }
    
    perp2 = edge_dir.cross(perp1).normalized();
    
    const float t = EDGE_THICKNESS;
    Vec3 corners[4] = {
        perp1 * t + perp2 * t,
        perp1 * t - perp2 * t,
        -perp1 * t - perp2 * t,
        -perp1 * t + perp2 * t
    };
    
    Vec3 start_verts[4], end_verts[4];
    for (int i = 0; i < 4; i++) {
        start_verts[i] = v0 + corners[i];
        end_verts[i] = v1 + corners[i];
    }
    
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        
        Polygon poly1;
        poly1.v0 = start_verts[i];
        poly1.v1 = end_verts[i];
        poly1.v2 = end_verts[j];
        poly1.computeNormal();
        poly1.color = EDGE_COLOR;
        poly1.object_id = object_id;
        poly1.reflection = 0.1f;
        poly1.transparency = 0.0f;
        poly1.is_floor = false;
        polygons.push_back(poly1);
        
        Polygon poly2;
        poly2.v0 = start_verts[i];
        poly2.v1 = end_verts[j];
        poly2.v2 = start_verts[j];
        poly2.computeNormal();
        poly2.color = EDGE_COLOR;
        poly2.object_id = object_id;
        poly2.reflection = 0.1f;
        poly2.transparency = 0.0f;
        poly2.is_floor = false;
        polygons.push_back(poly2);
    }
}

__host__ void createEdgeLightSource(const Vec3& v0, const Vec3& v1, const Vec3& center, 
                                   int object_id, int lights_per_edge, 
                                   std::vector<Polygon>& polygons, std::vector<Light>& lights) { 
    if (lights_per_edge <= 0) return;    
    for (int i = 1; i <= lights_per_edge; ++i) {
        float t = (float)i / (lights_per_edge + 1);
        Vec3 edge_point = v0 * (1.0f - t) + v1 * t;
        
        Vec3 edge_center = (v0 + v1) * 0.5f;
        Vec3 inward_dir = (center - edge_center).normalized();
        float inward_distance = 0.05f;
        
        Vec3 light_pos = edge_point + inward_dir * inward_distance;
        Light light;
        light.position = light_pos;
        light.color = WHITE_LIGHT * 10.0f;
        lights.push_back(light);

        const float r = 0.015f;
        std::vector<Vec3> sphere_vertices = {
            light_pos + Vec3(0, 0, r),
            light_pos + Vec3(r, 0, 0),
            light_pos + Vec3(0, r, 0),
            light_pos + Vec3(-r, 0, 0),
            light_pos + Vec3(0, -r, 0),
            light_pos + Vec3(0, 0, -r)
        };
        
        std::vector<Vec3i> sphere_triangles = {
            Vec3i(0, 1, 2), Vec3i(0, 2, 3), Vec3i(0, 3, 4), Vec3i(0, 4, 1),
            Vec3i(5, 2, 1), Vec3i(5, 3, 2), Vec3i(5, 4, 3), Vec3i(5, 1, 4)
        };
        
        for (const auto& tri : sphere_triangles) {
            Polygon light_poly;
            light_poly.v0 = sphere_vertices[tri.x];
            light_poly.v1 = sphere_vertices[tri.y];
            light_poly.v2 = sphere_vertices[tri.z];
            light_poly.computeNormal();

            light_poly.color = WHITE_LIGHT * 20.0f;
            light_poly.object_id = -2;
            light_poly.reflection = 0.0f;
            light_poly.transparency = 0.0f;
            light_poly.is_floor = false;
            
            polygons.push_back(light_poly);
        }
    }
}

__host__ void buildFigure(int id, const std::vector<Vec3>& base_vertices, const std::vector<Face>& base_faces, const Figure& fig, std::vector<Polygon>& polygons, std::vector<Light>& lights) {
    std::vector<Vec3> vertices;
    for (const auto& v : base_vertices) {
        vertices.push_back(v.normalized() * fig.radius + fig.center);
    }
    
    float glass_reflection = fig.reflection_coef;
    float glass_transparency = fig.transparent_coef;
    
    for (const auto& face : base_faces) {
        createFace(vertices, face, fig.color, id, polygons, glass_reflection, glass_transparency);
    }

    std::set<std::pair<int, int>> edges;
    for (const auto& face : base_faces) {
        const auto& indices = face.indices;
        for (size_t i = 0; i < indices.size(); i++) {
            int idx1 = indices[i];
            int idx2 = indices[(i + 1) % indices.size()];
            if (idx1 < idx2) {
                edges.insert(std::make_pair(idx1, idx2));
            } else {
                edges.insert(std::make_pair(idx2, idx1));
            }
        }
    }
    
    for (const auto& edge : edges) {
        int v0_idx = edge.first;
        int v1_idx = edge.second;
        
        if (v0_idx >= (int)vertices.size() || v1_idx >= (int)vertices.size())
            continue;
        
        createEdge(vertices[v0_idx], vertices[v1_idx], id, polygons);
    }

    if (fig.light_sources_number > 0) {
        int edge_count = 0;
        for (const auto& edge : edges) {
            
            int v0_idx = edge.first;
            int v1_idx = edge.second;
            
            if (v0_idx >= (int)vertices.size() || v1_idx >= (int)vertices.size())
                continue;
            
            Vec3 v0 = vertices[v0_idx];
            Vec3 v1 = vertices[v1_idx];
            
            createEdgeLightSource(v0, v1, fig.center, id, fig.light_sources_number, polygons, lights);
            
            edge_count++;
        }
    }
}

__host__ void buildFloor(const Floor& floor, std::vector<Polygon>& polygons) {
    Polygon floor1;
    floor1.v0 = floor.points[0];
    floor1.v1 = floor.points[1];
    floor1.v2 = floor.points[2];
    floor1.computeNormal();
    floor1.color = floor.color;
    floor1.object_id = -1;
    floor1.reflection = floor.reflection;
    floor1.transparency = 0.0f;
    floor1.is_floor = true;
    polygons.push_back(floor1);
    
    Polygon floor2;
    floor2.v0 = floor.points[0];
    floor2.v1 = floor.points[2];
    floor2.v2 = floor.points[3];
    floor2.computeNormal();
    floor2.color = floor.color;
    floor2.object_id = -1;
    floor2.reflection = floor.reflection;
    floor2.transparency = 0.0f;
    floor2.is_floor = true;
    polygons.push_back(floor2);
}

__host__ void buildScene(Scene& scene) {
    scene.polygons.clear();
    
    buildFloor(scene.floor, scene.polygons);
    
    std::vector<Vec3> tetra_vertices = createTetrahedronVertices();
    std::vector<Face> tetra_faces = createTetrahedronFaces();
    
    std::vector<Vec3> dodeca_vertices = createDodecahedronVertices();
    std::vector<Face> dodeca_faces = createDodecahedronFaces();
    
    std::vector<Vec3> icosa_vertices = createIcosahedronVertices();
    std::vector<Face> icosa_faces = createIcosahedronFaces();
    
    std::vector<Light> all_lights = scene.lights;
    
    for (int i = 0; i < (int)scene.figures.size() && i < 3; i++) {
        const Figure& fig = scene.figures[i];
        
        switch(fig.type) {
            case 0: // Тетраэдр
                buildFigure(i, tetra_vertices, tetra_faces, fig, scene.polygons, all_lights);
                break;
            case 1: // Додекаэдр
                buildFigure(i, dodeca_vertices, dodeca_faces, fig, scene.polygons, all_lights);
                break;
            case 2: // Икосаэдр
                buildFigure(i, icosa_vertices, icosa_faces, fig, scene.polygons, all_lights);
                break;
        }
    }
    
    scene.lights = all_lights;
}

#endif // __FIGURES_CUH__