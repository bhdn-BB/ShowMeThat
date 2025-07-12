import logging
import os


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'frames')
    TARGET_FORMAT_NOTE = '360p'
    MS_IN_SECOND = 1000
    IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')

    @staticmethod
    def get_logger(name: str = __name__) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger