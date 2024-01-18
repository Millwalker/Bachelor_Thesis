from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time=1
end_time=9
ffmpeg_extract_subclip("1person_close_front_p1.mp4", start_time, end_time, targetname="1person_close_front_p1_final.mp4")
