import os

from synthetic_cloth_data import DATA_DIR
from synthetic_cloth_data.meshes.utils.projected_mesh_area import get_mesh_projected_xy_area

mesh_dir = DATA_DIR / "flat_meshes" / "TSHIRT" / "Cloth3D-10"

mesh_paths = os.listdir(mesh_dir)
# filter out non-obj files
mesh_paths = [x for x in mesh_paths if x.endswith(".obj")]

import hashlib
import json
import os

if __name__ == "__main__":
    for filename in mesh_paths:

        # note that this is not the exact lattened area size the meshes are still  'fit around a human'..
        filename = str(filename)
        area = get_mesh_projected_xy_area(os.path.join(mesh_dir, filename))
        print(f"{filename}: {area}")

        with open(os.path.join(mesh_dir, filename.replace(".obj", ".json")), "r") as f:
            data = json.load(f)

        # save data to json file
        data.update(
            {
                "area": get_mesh_projected_xy_area(os.path.join(mesh_dir, filename)),
                "obj_md5_hash": hashlib.md5(open(os.path.join(mesh_dir, filename), "rb").read()).hexdigest(),
            }
        )
        with open(os.path.join(mesh_dir, filename.replace(".obj", ".json")), "w") as f:
            json.dump(data, f)

    print("done")
