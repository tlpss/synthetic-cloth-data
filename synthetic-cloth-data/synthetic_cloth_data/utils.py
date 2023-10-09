import datetime
import enum
import subprocess


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
    "left_waist",
    "right_waist",
    "right_pipe_outer",
    "right_pipe_inner",
    "crotch",
    "left_pipe_inner",
    "left_pipe_outer",
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


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_metadata_dict_for_dataset() -> dict:
    """dict with git hash and datetime for use in metadata files"""
    return {
        "git hash": get_git_revision_hash(),
        "date time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
