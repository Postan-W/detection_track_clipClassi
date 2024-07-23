from moviepy.editor import VideoFileClip, concatenate_videoclips
import glob

video_dir = "./videos/pose_infer/"
output_path = "videos/pose_infer/fall_merged2.mp4"

videos = glob.glob(video_dir+"*")


# 创建视频剪辑列表
video_clips = [VideoFileClip(video) for video in videos]

# 合并视频剪辑
final_clip = concatenate_videoclips(video_clips, method='compose')

# 输出合并后的视频文件
final_clip.write_videofile(output_path)