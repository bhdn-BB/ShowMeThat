import os
import shutil
import cv2
import yt_dlp as youtube_dl

OUTPUT_DIR = 'D:\\ShowMeThat\\ShowMeThat\\video_analyzer\\images'
TARGET_FORMAT_NOTE = '360p'
MS_IN_SECOND = 1000

def save_frames_from_video(
        video_url: str,
        frame_interval_sec: float | int
) -> None:
    print('Extracting video info...')
    try:
        youtube_dl_opts = {}
        with youtube_dl.YoutubeDL(youtube_dl_opts) as ydl:
            video_info = ydl.extract_info(video_url, download=False)
        formats = video_info.get('formats', None)

        if not formats:
            print('No video formats found.')
            return

        target_format = None
        for format_item in formats:
            if format_item.get('format_note') == TARGET_FORMAT_NOTE:
                target_format = format_item
                break

        if not target_format:
            print(f'Video format "{TARGET_FORMAT_NOTE}" not found.')
            return

        video_stream_url = target_format.get('url', None)
        video_capture = cv2.VideoCapture(video_stream_url)

        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        duration_ms = total_frames * (1/fps) * MS_IN_SECOND # in milliseconds

        current_time_ms = 0
        frame_index = 0

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        while video_capture.isOpened() and current_time_ms < duration_ms:
            video_capture.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
            read_success, current_frame = video_capture.read()

            if not read_success:
                print('Failed to read frame.')
                break

            frame_index += 1
            cv2.imwrite(
                os.path.join(
                    OUTPUT_DIR,
                    f"frame_{frame_index}_{video_url.split('=')[-1]}&t={round(current_time_ms/MS_IN_SECOND)}s.jpg"
                ),
                current_frame
            )
            current_time_ms += frame_interval_sec * MS_IN_SECOND # in milliseconds

        video_capture.release()

        print(f'Extracted {frame_index} frames.')

    except Exception as e:
        print(f'Failed to extract video info: {e}')
        return
    finally:
        print('Finished attempting to extract video info.')


# if __name__ == '__main__':
#
#     url = input('Enter video url:')
#     save_frames_from_video(url, 2)