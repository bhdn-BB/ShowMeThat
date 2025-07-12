import os
import cv2
import yt_dlp as youtube_dl
from search_service.config import Config


logger = Config.get_logger(__name__)

def save_frames_from_video(
        video_url: str,
        frame_interval_sec: float | int,
        quality: str = Config.TARGET_FORMAT_NOTE
) -> int | None:

    logger.info('Extracting video info...')
    try:
        youtube_dl_opts = {}
        with youtube_dl.YoutubeDL(youtube_dl_opts) as ydl:
            video_info = ydl.extract_info(video_url, download=False)

        video_id = video_info.get('id', 'unknown')
        formats = video_info.get('formats', None)

        if not formats:
            logger.warning('No video formats found.')
            return 0

        target_format = None
        for format_item in formats:
            if format_item.get('format_note') == quality:
                target_format = format_item
                break

        if not target_format:
            logger.warning(f'Video format "{quality}" not found.')
            return 0

        video_stream_url = target_format.get('url', None)
        if not video_stream_url:
            logger.warning('Stream URL not found.')
            return 0

        video_capture = cv2.VideoCapture(video_stream_url)

        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        duration_ms = total_frames * Config.MS_IN_SECOND / fps # in milliseconds

        current_time_ms = 0
        frame_index = 0

        # if os.path.exists(ConfigVideoProcessing.OUTPUT_DIR):
        #     shutil.rmtree(ConfigVideoProcessing.OUTPUT_DIR)
        # os.makedirs(ConfigVideoProcessing.OUTPUT_DIR)

        while video_capture.isOpened() or current_time_ms < duration_ms:

            video_capture.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
            read_success, current_frame = video_capture.read()


            if not read_success or current_frame is None:
                logger.warning(
                    f'Failed to read frame at {current_time_ms / Config.MS_IN_SECOND:.3f}s.'
                )
                break

            frame_index += 1

            filename = f"frame_{frame_index}_{video_id}&t={round(current_time_ms / Config.MS_IN_SECOND)}s.jpg"
            filepath = os.path.join(Config.OUTPUT_DIR, filename)
            cv2.imwrite(filepath, current_frame)

            current_time_ms += frame_interval_sec * Config.MS_IN_SECOND # in milliseconds

        video_capture.release()
        logger.info(f'Extracted {frame_index} frames.')
        return frame_index

    except Exception as e:
        logger.error(f'Failed to extract video info: {e}')
        return 0
    finally:
        logger.info('Finished attempting to extract video info.')

def build_youtube_link_from_filename(filename: str) -> str | None:
    try:
        # filename = "frame_3_n07rvqgxZfg&t=4s.jpeg"
        base = filename.split('.')[0]                # "frame_3_n07rvqgxZfg&t=4s"
        id_and_time = base.split('_', 2)[2]          # "n07rvqgxZfg&t=4s"
        link = f"https://www.youtube.com/watch?v={id_and_time}"
        logger.info(f'Generated link: {link}')
        return link
    except Exception as e:
        logger.error(f'Failed to build YouTube link from filename "{filename}": {e}')
        return None
    finally:
        logger.debug(f'Attempted to parse filename: {filename}')


# if __name__ == '__main__':
#
#     url = input('Enter video url:')
#     save_frames_from_video(url, 2)
    # build_youtube_link_from_filename('frame_2_IHOGcqGqdIQ&t=2s.jpg')
