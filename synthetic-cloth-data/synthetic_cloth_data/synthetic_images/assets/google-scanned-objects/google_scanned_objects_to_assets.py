"""Script that creates a blend file and loads all the Google Scanned Objects as blender assets.

run using blender -b -P <script> -- --path <path-to-google-scanned-objects-directory>

save this blend file in your asset directory and add it in Blender as an asset library.
"""
import os
import shutil
from typing import List, Tuple

import bpy
import numpy as np

# Requires Blender >=3.3 because obj import operator changed
# Also requires at least 22 GB of RAM + Swap space to pack the textures.


def get_field_value(field, txt, index):
    field = f"{field}: "
    find_index = txt.find(field, index)
    if find_index == -1:
        return None
    start = find_index + len(field) + 1
    end = txt.find('"', start)
    return txt[start:end]


def get_description_and_tags(model_directory) -> Tuple[str, List[str]]:
    metadata_path = os.path.join(model_directory, "metadata.pbtxt")

    tags = []

    with open(metadata_path, "r") as file:
        txt = file.read()

        # Get description
        description = ""
        lines = txt.splitlines()
        for line in lines:
            field = "description: "
            if line.startswith(field):
                description = line[len(field) + 1 : -1]  # noqa: E203
                break

        # Parse all annotations as tags
        n_annotations = txt.count("annotations")
        index = 0
        for _ in range(n_annotations):
            index = txt.find("annotations", index) + len("annotations")
            key = get_field_value("key", txt, index)
            value = get_field_value("value", txt, index)
            tag = f"{key}: {value}"
            tags.append(tag)

        # Parse the category as a tag
        n_categories = txt.count("categories")
        index = 0
        for _ in range(n_categories):
            index = txt.find("categories", index) + len("categories")
            category = get_field_value("first", txt, index)
            if category is None:
                category = "Uncategorized"
            tags.append(category)

    return description, tags


def move_texture_png(model_directory):
    texture_path = os.path.join(model_directory, "materials", "textures", "texture.png")
    texture_destination_path = os.path.join(model_directory, "meshes", "texture.png")
    if not os.path.exists(texture_destination_path):
        # copy instead of move so that the original directory is not modified
        shutil.copy(texture_path, texture_destination_path)


def load_gso_model(model_directory):
    move_texture_png(model_directory)
    obj_path = os.path.join(model_directory, "meshes", "model.obj")
    bpy.ops.object.select_all(action="DESELECT")
    # bpy.ops.import_scene.obj(filepath=obj_path, up_axis="Z") #, split_mode="OFF")
    bpy.ops.wm.obj_import(filepath=obj_path, up_axis="Z")
    object = bpy.context.selected_objects[0]
    return object


def all_gso_downloads_to_assets(gso_directory):
    model_directories = [f.path for f in os.scandir(gso_directory) if f.is_dir()]
    model_directories = model_directories

    print(len(model_directories))

    n = len(model_directories)
    columns = int(np.ceil(np.sqrt(n)))
    spacing = 0.55  # The scanner capture area is roughly 50cm

    print(f"Found {n} models, using {columns} columns.")

    shared_tags = ["GSO", "Google", "Google Scanned Objects"]

    for i, model_directory in enumerate(model_directories):
        print(i, model_directory)
        x = spacing * (i % columns)
        y = -spacing * (i // columns)
        object = load_gso_model(model_directory)
        object.location = (x, y, 0)
        object.name = os.path.basename(model_directory)
        object.asset_mark()

        thumbnail = os.path.join(model_directory, "thumbnails", "0.jpg")

        override = bpy.context.copy()
        override["id"] = object
        with bpy.context.temp_override(**override):
            bpy.ops.ed.lib_id_load_custom_preview(filepath=thumbnail)

        object.asset_data.author = "Google"
        description, metadata_tags = get_description_and_tags(model_directory)
        object.asset_data.description = description
        tags = shared_tags + metadata_tags
        for tag in tags:
            object.asset_data.tags.new(tag)

        bpy.ops.file.pack_all()


if __name__ == "__main__":
    import sys

    bpy.ops.object.delete()

    # check if id was passed as argument
    gso_directory = ""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        gso_directory = argv[argv.index("--path") + 1]

    print(gso_directory)
    if not gso_directory:
        print("Please pass the path to the GSO models as an argument")
        exit()

    if not os.path.exists(gso_directory):
        print(f"Please download the GSO models to {gso_directory}")
        exit()

    # all_gso_downloads_to_assets(gso_directory)

    blend_file_path = os.path.join(gso_directory, "GSO.blend")
    # bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)

    bpy.ops.preferences.asset_library_add(directory=gso_directory)
    new_library = bpy.context.preferences.filepaths.asset_libraries[-1]
    new_library.name = "Google Scanned Objects"
    bpy.ops.wm.save_userpref()

    print("done")