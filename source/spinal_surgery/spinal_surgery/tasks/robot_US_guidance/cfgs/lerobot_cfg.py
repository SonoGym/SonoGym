GUIDANCE_TASK = "Open a dishwasher door completely while switching contact points."
GUIDANCE_FEATURES = {
    "observation.images": {
        "dtype": "image",
        "shape": (200, 150, 3),
        "names": [
            "height",
            "width",
            "channel",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": [
            "empty"
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (3,),
        "names": {
            "axes": [
                "tip_frame_x",
                "tip_frame_y",
                "tip_frame_z_angle",
            ],
        },
    },
}