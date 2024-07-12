import subprocess
import os


input_video = "./main_code/shuaidao/videos/2.mp4"
output_video = "./main_code/shuaidao/videos/2.mp4"


def video_trimming(input_video, output_video, start_time, end_time):
    # subprocess.call(['ffmpeg', '-i', input_video, '-ss', str(start_time), '-to', str(end_time),output_video])
    os.system("ffmpeg -i {} -ss {} -to {} {}".format(input_video,start_time,end_time,output_video))
video_trimming(input_video, output_video, "00:03:20","00:03:50")