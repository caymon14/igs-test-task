import logging
from pathlib import Path
import argparse
from app.video_processor import process_video

logger = logging.getLogger()


def main(args):
    try:
        results = process_video(video_path=Path(args.video))
        if results:
            print("Results:")
            for number, plate_image in results:
                if len(number) > 5:
                    print(number)
                else:
                    print(f"Incorrect number: {number}")
        else:
            print("Numbers was not found")
    except Exception as ex:
        logger.fatal(ex, exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, help="Path to video for recognition.")
    args = parser.parse_args()
    main(args)
