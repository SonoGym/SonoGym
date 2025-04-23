SURGERY_TASK = "perform spine surgery based on ultrasound volume."
SURGERY_FEATURES = {
    "observation.images": {
        "dtype": "image",
        "shape": (250, 185, 3),
        "names": [
            "height",
            "width",
            "channel",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": [
                "US_to_tip_x",
                "US_to_tip_y",
                "US_to_tip_z",
                "US_to_tip_q_w",
                "US_to_tip_q_x",
                "US_to_tip_q_y",
                "US_to_tip_q_z",
            ],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": [
                "tip_frame_x",
                "tip_frame_y",
                "tip_frame_z",
                "tip_frame_angle_axis_x",
                "tip_frame_angle_axis_y",
                "tip_frame_angle_axis_z",
            ],
        },
    },
}