import enum

CLOTH_TYPES = enum.Enum("CLOTH_TYPES", "TOWEL SHORTS TSHIRT")

CLOTH_TYPE_TO_COCO_CATEGORY_ID = {
    CLOTH_TYPES.TOWEL.name: 0,
    CLOTH_TYPES.SHORTS.name: 1,
    CLOTH_TYPES.TSHIRT.name: 2,
}
TOWEL_KEYPOINTS = [
    "corner_0",
    "corner_1",
    "corner_2",
    "corner_3",
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
