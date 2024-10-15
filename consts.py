detection_model_path = "detection_model.pt"  # "yolov8-n-31.05.24.pt" "detection_model.pt"
color_model_path = "efficientnet_b3_0_992.pth"

color_map = {
    0: 'yellow',
    1: 'blue_str',
    2: 'red_str',
    3: 'pink_str',
    4: 'orange_str',
    5: 'green_str',
    6: 'brown_str',
    7: 'blue',
    8: 'red',
    9: 'pink',
    10: 'orange',
    11: 'green',
    12: 'brown',
    13: 'black',
    14: 'yellow_str',
    15: 'cue_ball',
}

number_map = {
    'yellow': 1,
    'blue_str': 10,
    'red_str': 11,
    'pink_str': 21,
    'orange_str': 13,
    'green_str': 14,
    'brown_str': 15,
    'blue': 2,
    'red': 3,
    'pink': 4,
    'orange': 5,
    'green': 6,
    'brown': 7,
    'black': 8,
    'yellow_str': 9,
    'cue_ball': 'cue_ball',
}