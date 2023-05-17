"""script that downloads the PolyHaven HDRIs in the desired resolution

This is a quick fix, instead of changing the resoluation at runtime using the add-ons 'switch Resolution' operator
because the operator is 'modal' and the feedback mechanism was not working
so by downloading them upfront, the switch_resolution operation works instantly and the
renders will already use the HDRI in the desired resolution.
"""

import json
import os

import bpy
import requests
import tqdm

if __name__ == "__main__":
    resolution = "4k"

    polyhaven_assets_path = bpy.context.preferences.filepaths.asset_libraries["Poly Haven"].path
    print(polyhaven_assets_path)

    assets = os.listdir(polyhaven_assets_path)
    # only keep the directories, which actually contain the assets
    assets = [asset for asset in assets if os.path.isdir(polyhaven_assets_path + "/" + asset)]

    for asset in tqdm.tqdm(assets):
        json_path = polyhaven_assets_path + "/" + asset + "/info.json"
        info_dict = json.load(open(json_path, "r"))
        if info_dict["type"] != 0:  # HDRI
            continue
        if "indoor" not in info_dict["categories"]:
            # only download indoor HDRI's
            continue
        hdri_url = info_dict["files"]["hdri"][resolution]["hdr"]["url"]
        print(hdri_url)
        # download the hdri file into the folder

        header = requests.utils.default_headers()
        # avoid 403 error by providing 'User-Agent' header
        # https://github.com/Poly-Haven/polyhavenassets/blob/259c0a96a70414ab3d1943c720d885e471aac192/constants.py#L11
        header.update({"User-Agent": "Blender: PH Assets"})
        result = requests.get(hdri_url, headers=header)
        with open(json_path.replace("info.json", hdri_url.split("/")[-1]), "wb") as file:
            file.write(result.content)
