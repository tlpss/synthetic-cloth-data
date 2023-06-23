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
        g_params.radius = 0.0175f;
        g_numExtraParticles = 20000;
        g_params.collisionDistance = g_params.radius*0.5f;

        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_params.dynamicFriction = 0.75f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;

        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;

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
        }



    }


};