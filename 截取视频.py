import subprocess
import os


input_video = "./videos/fanyue/real_negative111.mp4"
output_video = "./videos/fanyue_negative_trimmed.mp4"


def video_trimming(input_video, output_video, start_time, end_time):
    # subprocess.call(['ffmpeg', '-i', input_video, '-ss', str(start_time), '-to', str(end_time),output_video])
    os.system("ffmpeg -i {} -ss {} -to {} {}".format(input_video,start_time,end_time,output_video))
video_trimming(input_video, output_video, "00:00:00","00:00:30")