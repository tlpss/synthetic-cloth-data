import json
import pathlib

import airo_blender as ab

POLYHAVEN_ASSETS_SNAPSHOT_PATH = pathlib.Path(__file__).parent / "polyhaven_assets_snapshot.json"
if __name__ == "__main__":
    all_assets = ab.available_assets()
    polyhaven_assets = [asset for asset in all_assets if asset["library"] == "Poly Haven"]

    asset_snapshot = {"assets": polyhaven_assets}

    with open(POLYHAVEN_ASSETS_SNAPSHOT_PATH, "x") as file:
        json.dump(asset_snapshot, file, indent=4)
