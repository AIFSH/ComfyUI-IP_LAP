import os
import platform
import subprocess
import folder_paths
from pydub import AudioSegment
from moviepy.editor import VideoFileClip,AudioFileClip

parent_directory = os.path.dirname(os.path.abspath(__file__))

from .ip_lap.inference import IP_LAP_infer

input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class CombineAudioVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"vocal_AUDIO": ("AUDIO",),
                     "bgm_AUDIO": ("AUDIO",),
                     "video": ("VIDEO",)
                    }
                }

    CATEGORY = "AIFSH_IP_LAP"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "combine"

    def combine(self, vocal_AUDIO,bgm_AUDIO,video):
        vocal = AudioSegment.from_file(vocal_AUDIO)
        bgm = AudioSegment.from_file(bgm_AUDIO)
        audio = vocal.overlay(bgm)
        audio_file = os.path.join(out_path,"ip_lap_voice.wav")
        audio.export(audio_file, format="wav")
        cm_video_file = os.path.join(out_path,"voice_"+os.path.basename(video))
        video_clip = VideoFileClip(video)
        audio_clip = AudioFileClip(audio_file)
        new_video_clip = video_clip.set_audio(audio_clip)
        new_video_clip.write_videofile(cm_video_file)
        return (cm_video_file,)


class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_IP_LAP"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}
    
    CATEGORY = "AIFSH_IP_LAP"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO","AUDIO")

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_path,video)
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.join(input_path,video+".wav")
        video_clip.audio.write_audiofile(audio_path)
        return (video_path,audio_path,)

class IP_LAP:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "audio": ("AUDIO",),
                        "video": ("VIDEO",),
                        "T":("INT",{
                            "default": 5,
                        }),
                        "Nl":("INT",{
                            "default": 15,
                        }),
                        "ref_img_N":("INT",{
                            "default": 25,
                        }),
                        "img_size":("INT",{
                            "default": 128,
                        }),
                        "mel_step_size":("INT",{
                            "default": 16,
                        }),
                        "face_det_batch_size":("INT",{
                            "default": 4,
                        }),
                        "checkpoints_path":("STRING",{
                            "default": os.path.join(parent_directory,"weights")
                        })
                    }
                }

    CATEGORY = "AIFSH_IP_LAP"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "process"

    def process(self, audio, video, T=5,Nl=15,ref_img_N=25,img_size=128,
                mel_step_size=16,face_det_batch_size=4,checkpoints_path=""):
        ip_lap = IP_LAP_infer(T,Nl,ref_img_N,img_size,mel_step_size,face_det_batch_size,checkpoints_path)
        video_name = os.path.basename(video)
        out_video_file = os.path.join(out_path, f"ip_lap_{video_name}")
        ip_lap(video,audio,out_video_file)
        # res_video_file = os.path.join(out_path, f"result_ip_lap_{video_name}")
        # command = f'ffmpeg -y -i {out_video_file} -i {audio} -map 0:0 -map 1:0 -c:a libmp3lame -q:a 1 -q:v 1 -shortest {res_video_file}'
        # subprocess.call(command, shell=platform.system() != 'Windows')
        return (out_video_file,)