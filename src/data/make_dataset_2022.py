from pathlib import Path
import logging

logging.basicConfig(
    filename="dicom_conversion.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

import click
import pandas as pd
from okapy.dicomconverter.converter import NiftiConverter

from src.data.utils import (correct_names, move_extra_vois, clean_vois,
                            compute_bbs)

project_dir = Path(__file__).resolve().parents[2]
# default_input_path = project_dir / "hecktor/data/hecktor2022/raw/mda_test"
default_input_path = "/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/dicom/mda_test"
default_images_folder = project_dir / "data/hecktor2022/processed/mda_test/images"
default_labels_folder = project_dir / "data/hecktor2022/processed/mda_test/labels"
default_dump = project_dir / "data/hecktor2022/processed/mda_test/labels"
default_name_mapping = project_dir / "data/hecktor2021_name_mapping_testing.csv"


@click.command()
@click.argument('input_folder', type=click.Path(), default=default_input_path)
@click.argument('output_images_folder',
                type=click.Path(),
                default=default_images_folder)
@click.argument('output_labels_folder',
                type=click.Path(),
                default=default_labels_folder)
@click.argument('dump_folder', type=click.Path(), default=default_dump)
@click.option("--name_mapping",
              type=click.Path(),
              default=default_name_mapping)
def main(input_folder, output_images_folder, output_labels_folder, dump_folder,
         name_mapping):
    """Command Line Interface to make the dataset for the HECKTOR Challenge
        In short, this routine convert the DICOM files to NIFTI and stores the CT in 
        Hounsfield Unit and the PET in Standardized Uptake value.
    """

    output_images_folder = Path(output_images_folder)
    output_labels_folder = Path(output_labels_folder)
    dump_folder = Path(dump_folder)
    output_images_folder.mkdir(exist_ok=True, parents=True)
    dump_folder.mkdir(exist_ok=True, parents=True)
    logger.info("Converting Dicom to Nifty - START")
    converter = NiftiConverter(
        padding="whole_image",
        labels_startswith="GTV",
        cores=24,
        naming=2,
    )
    _ = converter(input_folder, output_folder=output_images_folder)

    logger.info("Converting Dicom to Nifty - END")
    # logger.info("Removing extra VOI - START")
    # move_extra_vois(output_images_folder, archive_folder)
    # logger.info("Removing extra VOI - END")
    # logger.info("Renaming files- START")
    # correct_names(output_images_folder, name_mapping)
    # logger.info("Renaming files- END")
    # logger.info("Cleaning the VOIs - START")
    # clean_vois(output_images_folder)
    # logger.info("Cleaning the VOIs - END")


if __name__ == '__main__':
    main()