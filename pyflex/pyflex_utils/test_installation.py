import numpy as np

import pyflex


def get_default_config(
    particle_radius=0.0175,
    cloth_stiffness=(0.75, 0.02, 0.02),
    scale=1.0,
):
    config = {
        "scale": scale,
        "cloth_pos": [0.0, 1.0, 0.0],
        "cloth_size": [int(0.6 / particle_radius), int(0.368 / particle_radius)],
        "cloth_stiff": cloth_stiffness,  # Stretch, Bend and Shear
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "render_type": ["cloth"],
                "cam_position": [0, 5, 0],
                "cam_angle": [np.pi / 2, -np.pi / 2, 0],
                "cam_size": [480, 480],
                "cam_fov": 39.5978 / 180 * np.pi,
            }
        },
        "scene_config": {
            "scene_id": 0,
            "radius": particle_radius * scale,
            "buoyancy": 0,
            "numExtraParticles": 20000,
            "collision_distance": 0.0006,
            "particle_radius": 0.004,
            "solid_rest_distance": 0.004,
            "msaaSamples": 0,
        },
        "flip_mesh": 0,
    }

    return config


if __name__ == "__main__":
    pyflex.init(True, False, 480, 480, 0)
    config = get_default_config()
    pyflex.set_scene(0, config["scene_config"])
