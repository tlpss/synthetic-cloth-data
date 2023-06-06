"""script to create asset snapshots (json files with metadata about the assets). These are used to sample assets for the synthetic images.

Usage:
blender -P -b <path_to_this_file>

Make sure to add (and create if required) the asset files first
"""
import json

import airo_blender as ab
from synthetic_cloth_data.synthetic_images.assets.asset_snapshot_paths import (
    GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_PATH,
    POLYHAVEN_ASSETS_SNAPSHOT_PATH,
)


def create_asset_json(assets, snapshot_path):
    asset_snapshot = {"assets": assets}

    with open(snapshot_path, "w") as file:
        json.dump(asset_snapshot, file, indent=4)


if __name__ == "__main__":
    all_assets = ab.available_assets()

    polyhaven_assets = [asset for asset in all_assets if asset["library"] == "Poly Haven"]
    print(f"Found {len(polyhaven_assets)} polyhaven assets")
    create_asset_json(polyhaven_assets, POLYHAVEN_ASSETS_SNAPSHOT_PATH)

    gso_assets = [asset for asset in all_assets if asset["library"] == "Google Scanned Objects"]
    print(f"Found {len(gso_assets)} google scanned objects assets")
    create_asset_json(gso_assets, GOOGLE_SCANNED_OBJECTS_ASSETS_SNAPSHOT_PATH)

    print("Done!")
