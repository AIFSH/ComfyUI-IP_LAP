from .nodes import IP_LAP,LoadVideo,PreViewVideo,CombineAudioVideo
WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "IP_LAP": IP_LAP,
    "LoadVideo": LoadVideo,
    "PreViewVideo": PreViewVideo,
    "CombineAudioVideo": CombineAudioVideo
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "IP_LAP": "IP_LAP Node",
    "LoadVideo": "Video Loader",
    "PreViewVideo": "PreView Video",
    "CombineAudioVideo": "Combine Audio Video"
}
