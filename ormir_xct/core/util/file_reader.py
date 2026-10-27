"""
Created by: Michael Kuczynski
Created on: June 18th, 2022
"""
from pathlib import Path
from typing_extensions import deprecated

import SimpleITK as sitk

from ormir_mids.utils.io import load_dicom, \
                                load_dicom_with_subfolders, \
                                load_omids, \
                                load_scanco, \
                                scanco_file_extensions

from ormir_mids.utils import OMidsMedVolume

file_extensions = [".nii", ".nii.gz", ".mha", ".nrrd"] + scanco_file_extensions


def file_reader(input_file_path):
    """
    Read the input file using the correct reader.

    Parameters
    ----------
    input_file_path : string or pathlib.Path
        Path to the input image file.
    
    Returns
    -------
    image : SimpleITK.Image, OMidsMedVolume, tuple, or numpy.ndarray
    """
    if not isinstance(input_file_path, Path):
        input_file_path = Path(input_file_path)

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file {input_file_path} does not exist.")

    if not input_file_path.is_file():
        # assume it's a directory containing a DICOM series
        if not any(input_file_path.iterdir()):
            raise FileNotFoundError(f"Input directory {input_file_path} is empty.")

        if any(subdir.is_dir() for subdir in input_file_path.iterdir()):
            # If there are subdirectories, use load_dicom_with_subfolders
            image = load_dicom_with_subfolders(str(input_file_path))
        else:
            # If there are no subdirectories, use load_dicom
            image = load_dicom(str(input_file_path))
    else:
        # Get the file extension and check if we can read it
        input_filename = input_file_path.name
        input_extension = input_file_path.suffix

        input_extension = input_extension.split(";")[0].lower() # Handle Scanco files with version numbers in the extension

        if input_extension not in file_extensions:
            raise ValueError(f"File extension {input_extension} not supported.\nSupported extensions are: {file_extensions}")

        if input_extension in scanco_file_extensions:
            # Handle scanco files
            image = load_scanco(str(input_file_path))
        elif input_extension in [".nii", ".nii.gz"]:
            image = load_omids(str(input_file_path))
        else:
            # Handle MHA and NRRD files
            image = sitk.ReadImage(str(input_file_path))

    # Convert to the requested type if specified
    # if type is not None:
    #     image = convert_to_type(image, type)
    # else:
    #     # Default to returning an OMidsMedVolume object
    #     image = convert_to_type(image, OMidsMedVolume)

    return image


@deprecated("verify_image is deprecated and will be removed in a future release. Use file_reader instead.")
def verify_image(input_image, precision=None):
    """
    Provide with string path or sitk.Image. Ensures string paths are converted to sitk.Image before processing.
    Avoid using for masks as the file_reader method uses sitk.sitkFloat32 precision.

    Parameters
    ----------
    input_image : string path to an sitk readable image or an sitk.Image
    precision: set precision if desired, None by default

    Returns
    -------
    image : SimpleITK.Image
    """

    if isinstance(input_image, str):
        return file_reader(input_image, sitk.Image)
    else:
        return sitk.Cast(input_image, precision) if precision else input_image
