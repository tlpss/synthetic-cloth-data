# Blender Asset setup

This readme describes how to gather all the assets for the synthetic data generation. It requires a working `airo-blender` setup.
To store all the assets, we recommend to create a `blender-assets` folder somewhere. Make sure you have enough space, as the assets will take 20GB or more.

## PolyHaven
- Install the polyhaven addon and download all the available assets, see [here](https://github.com/Poly-Haven/polyhavenassets) for instructions. This cannot be done headless and hence requires a workstation with a monitor (or screen forwarding). The suggested locatation for the assets is a directory called `polyhaven` in your `blender-assets` directory.
- run the polyhaven HDRI downloader script `polyhaven/polyhaven_hdri_downloader.py`, which will download all the HDRIs in the desired resolution.

## Google Scanned Objects
- run the download script to download the assets to the desired location.
- run the script to import the GSOs to blender
- save the blender file with all the assets into the directory where you downloaded the GSOs
- Add an Asset Library to blender (preferences > file paths) that points to the directory of the GSO and name it`Google Scanned Objects`
- save and you are done

## Snapshots
- run the `make_assets_snapshots.py` file to obtain a json for each asset library. The main purpose is to enable reproducibility, both over time on the same machine and for different users/machines.



