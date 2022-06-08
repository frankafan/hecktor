from src.data.utils import compute_bbs
from pathlib import Path
import logging

project_dir = Path(__file__).resolve().parents[2]
default_images_folder = project_dir / "data/hecktor2021/hecktor_nii/"
default_bbox_path = project_dir / "data/hecktor2021/bbox.csv"

logging.basicConfig(
    filename="generate_bbox.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info("Computing the bounding boxes - START")
    bb_df = compute_bbs(default_images_folder)
    bb_df.to_csv(default_bbox_path)
    logger.info("Computing the bounding boxes - END")
