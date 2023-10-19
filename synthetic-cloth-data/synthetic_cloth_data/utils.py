import datetime
import enum
import subprocess

"""single point of truth for the cloth types and their coco category ids, the keypoint names and their order within each category. Should match the aRTFClothes dataset"""


class CLOTH_TYPES(str, enum.Enum):
    LEGO = "LEGO"  # LEGO battery object for testing
    TOWEL = "TOWEL"
    SHORTS = "SHORTS"
    TSHIRT = "TSHIRT"


CLOTH_TYPE_TO_COCO_CATEGORY_ID = {
    CLOTH_TYPES.LEGO.name: 444,
    CLOTH_TYPES.TOWEL.name: 0,
    CLOTH_TYPES.SHORTS.name: 1,
    CLOTH_TYPES.TSHIRT.name: 2,
}
TOWEL_KEYPOINTS = [
    "corner0",
    "corner1",
    "corner2",
    "corner3",
]

SHORT_KEYPOINTS = [
    "waist_left",
    "waist_right",
    "pipe_right_outer",
    "pipe_right_inner",
    "crotch",
    "pipe_left_inner",
    "pipe_left_outer",
]

TSHIRT_KEYPOINTS = [
    "shoulder_left",
    "neck_left",
    "neck_right",
    "shoulder_right",
    "sleeve_right_top",
    "sleeve_right_bottom",
    "armpit_right",
    "waist_right",
    "waist_left",
    "armpit_left",
    "sleeve_left_bottom",
    "sleeve_left_top",
]

LEGO_KEYPOINTS = ["knob", "top-right", "top-left", "center-front", "center-back"]

CATEGORY_NAME_TO_KEYPOINTS_DICT = {
    CLOTH_TYPES.LEGO.name: LEGO_KEYPOINTS,
    CLOTH_TYPES.TOWEL.name: TOWEL_KEYPOINTS,
    CLOTH_TYPES.SHORTS.name: SHORT_KEYPOINTS,
    CLOTH_TYPES.TSHIRT.name: TSHIRT_KEYPOINTS,
}


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_metadata_dict_for_dataset() -> dict:
    """dict with git hash and datetime for use in metadata files"""
    return {
        "git hash": get_git_revision_hash(),
        "date time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
