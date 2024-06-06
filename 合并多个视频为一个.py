from moviepy.editor import VideoFileClip, concatenate_videoclips
import glob

video_dir = ""
video_list =


# 创建视频剪辑列表
video_clips = [VideoFileClip(video) for video in video_files]

# 合并视频剪辑
final_clip = concatenate_videoclips(video_clips, method='compose')

output_path = ""
# 输出合并后的视频文件
final_clip.write_videofile('merged_video.mp4')