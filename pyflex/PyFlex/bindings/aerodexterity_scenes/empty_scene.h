#pragma once
#include <iostream>
#include <vector>

class EmptyScene : public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;

    EmptyScene(const char *name) : Scene(name) {}

    void Initialize(py::dict scene_params)
    {
        std::cout << "Initialize EmptyScene" << std::endl;
        g_drawPoints = false;
        g_drawCloth = false;
        g_drawSprings = false;

        // set default values
        g_numExtraParticles = 20000;

        g_params.radius = 0.01f; // interaction radius: threshold radius for particle-particle interactions
        g_params.collisionDistance = 0.003f; // distance for particle/shape collision detection, should be >0 to avoid 'tunneling'
        g_params.solidRestDistance = 0.01f; // distance between particles in rest, should approx match edge lengths of the cloth mesh and determines 'thickness' as well.

        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_params.dynamicFriction = 0.75f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;

        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;

        g_params.drag = 0.0001f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        for (auto item : scene_params){
            string key = py::str(item.first);
            //std::cout << "key: " << key << std::endl;
            if (key == "dynamic_friction"){
                g_params.dynamicFriction = std::stof(py::str(item.second));
                //std::cout << "dynamic_friction: " << g_params.dynamicFriction << std::endl;
            }
            if (key == "particle_friction") g_params.particleFriction = std::stof(py::str(item.second));
            if (key == "static_friction") g_params.staticFriction = std::stof(py::str(item.second));
            if (key == "drag") g_params.drag = std::stof(py::str(item.second));
            if (key == "particle_radius") g_params.radius = std::stof(py::str(item.second));
            if (key == "collision_distance") g_params.collisionDistance = std::stof(py::str(item.second));
            if (key == "solid_rest_distance") g_params.solidRestDistance = std::stof(py::str(item.second));
        }

        std::cout << "scene parameters:" << std::endl;
        std::cout << "g_params.dynamicFriction: " << g_params.dynamicFriction << std::endl;
        std::cout << "g_params.particleFriction: " << g_params.particleFriction << std::endl;
        std::cout << "g_params.staticFriction: " << g_params.staticFriction << std::endl;
        std::cout << "g_params.drag: " << g_params.drag << std::endl;
        std::cout << "g_params.radius: " << g_params.radius << std::endl;
        std::cout << "g_params.collisionDistance: " << g_params.collisionDistance << std::endl;
        std::cout << "g_params.solidRestDistance: " << g_params.solidRestDistance << std::endl;



    }


};