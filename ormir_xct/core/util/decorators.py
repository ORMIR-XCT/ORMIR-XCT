"""
Created by:   Johannes Bereiter-Payr
Created on:   Sept 29, 2026

Description: Decorator functions to handle image type conversions for image processing functions.
"""

import functools
import copy

import SimpleITK as sitk
import numpy as np

from pathlib import Path

from .file_reader import file_reader

from ormir_mids.utils.OMidsMedVolume import MedicalVolume, VoxelMedicalVolume
from ormir_mids.utils.headers import remerge_headers, separate_headers
from ormir_mids.utils.io import numpy_image_tuple

__all__ = [
    "request_sitk",
    "request_numpy",
    "request_medicalvolume",
    "request_tuple",
    "convert_to_type",
]


def _get_sitk_metadata(image):
    """
    Extract metadata from a SimpleITK image and return a deep copy.

    Parameters
    ----------
    image : SimpleITK.Image

    Returns
    -------
    metadata : dict
        Dictionary containing the metadata of the image.
    """
    metadata = {}
    for key in image.GetMetaDataKeys():
        metadata[key] = copy.deepcopy(image.GetMetaData(key))
    return metadata


def _convert_to_sitk_image(image, accept_none=True):
    """
    Convert an image to SimpleITK.Image if it is not already.

    Parameters
    ----------
    image : SimpleITK.Image, OMidsMedVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    sitk_image : SimpleITK.Image
    """
    if image is None:
        if accept_none:
            return None
        else:
            raise TypeError("NoneType is not a valid image.")
        
    if isinstance(image, str) or isinstance(image, Path):
        # If the input is a string (file path), read the image using file_reader
        image = file_reader(image)  # Read the image and convert to SimpleITK.Image
    
    if isinstance(image, sitk.Image):
        return image
    elif isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        im = sitk.GetImageFromArray(image[0])
        meta = image[1]
        for k, v in meta.items():
            im.SetMetaData(k, copy.deepcopy(v))
        return im
    elif isinstance(image, (MedicalVolume, VoxelMedicalVolume)):
        return image.to_sitk()
    elif isinstance(image, np.ndarray):
        return sitk.GetImageFromArray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected SimpleITK.Image, OMidsMedVolume, or numpy.ndarray.")


def _convert_to_medicalvolume(image, accept_none=True):
    """
    Convert an image to MedicalVolume if it is not already.

    Parameters
    ----------
    image : SimpleITK.Image, MedicalVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    medicalvolume : MedicalVolume
    """
    if image is None:
        if accept_none:
            return None
        else:
            raise TypeError("NoneType is not a valid image.")
    
    if isinstance(image, str) or isinstance(image, Path):
        # If the input is a string (file path), read the image using file_reader
        image = file_reader(image)  # Read the image and convert to MedicalVolume

    if isinstance(image, (MedicalVolume, VoxelMedicalVolume)):
        return image
    elif isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        omidsmed = MedicalVolume(image[0], np.eye(4))
        header = image[1] if isinstance(image[1], dict) else {}
        if header:
            omidsmed.omids_header, omidsmed.patient_header, omidsmed.extra_header = separate_headers(header)
        else:
            omidsmed.omids_header = {}
            omidsmed.patient_header = {}
            omidsmed.extra_header = {}
        return omidsmed
    elif isinstance(image, sitk.Image):
        return MedicalVolume.from_sitk(image)
    elif isinstance(image, np.ndarray):
        return MedicalVolume(image, np.eye(4))
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected SimpleITK.Image.")


def _convert_to_numpy_array(image, accept_none=True):
    """
    Convert an image to numpy.ndarray if it is not already. This function will return the raw data of the image, ignoring any metadata.

    Parameters
    ----------
    image : SimpleITK.Image, MedicalVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    numpy_array : numpy.ndarray
    """
    if image is None:
        if accept_none:
            return None
        else:
            raise TypeError("NoneType is not a valid image.")
        
    if isinstance(image, str) or isinstance(image, Path):
        # If the input is a string (file path), read the image using file_reader
        image = file_reader(image, type=np.ndarray)  # Read the image and convert to numpy.ndarray
    
    if isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        return image[0]  # Already in the desired 
    elif isinstance(image, sitk.Image):
        return sitk.GetArrayFromImage(image)
    elif isinstance(image, (MedicalVolume, VoxelMedicalVolume)):
        return np.asarray(image.A)  # Convert MedicalVolume to numpy array
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected str, pathlib.Path, SimpleITK.Image, MedicalVolume, tuple or numpy.ndarray.")


def _convert_to_tuple(image, accept_none=True):
    """
    Convert an image to a tuple of (numpy.ndarray, metadata_array) if it is not already.

    Parameters
    ----------
    image : SimpleITK.Image, MedicalVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray

    Returns
    -------
    tuple : (numpy.ndarray, metadata_array)
        A tuple containing the image data as a numpy array and the metadata.
    """
    if image is None:
        if accept_none:
            return None
        else:
            raise TypeError("NoneType is not a valid image.")

    if isinstance(image, str) or isinstance(image, Path):
        # If the input is a string (file path), read the image using file_reader
        image = file_reader(image, type=tuple)  # Read the image and convert to tuple format
        
    if isinstance(image, tuple) and len(image) == 2 and isinstance(image[0], np.ndarray):
        return image  # Already in the desired format
    elif isinstance(image, sitk.Image):
        return numpy_image_tuple(sitk.GetArrayFromImage(image), _get_sitk_metadata(image))
    elif isinstance(image, (MedicalVolume, VoxelMedicalVolume)):
        return numpy_image_tuple(image.A, remerge_headers(image.omids_header, image.patient_header, image.extra_header))
    elif isinstance(image, np.ndarray):
        return numpy_image_tuple(image, {})  # No metadata information available for raw numpy arrays
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected str, pathlib.Path, SimpleITK.Image, MedicalVolume, tuple or numpy.ndarray.")


def convert_to_type(image, target_type, accept_none=True):
    """
    Convert an image to the specified target type.

    Parameters
    ----------
    image : SimpleITK.Image, MedicalVolume, tuple of (numpy.ndarray, dict) or numpy.ndarray
    target_type : class
        The desired output type. Can be SimpleITK.Image, MedicalVolume, tuple or numpy.ndarray.

    Returns
    -------
    converted_image : target_type
        The image converted to the specified target type.
    """
    if image is None:
        if accept_none:
            return None
        else:
            raise TypeError("NoneType is not a valid image.")
        
    if isinstance(image, target_type):
        return image  # Already in the desired format
    if target_type == tuple:
        return _convert_to_tuple(image, accept_none=accept_none)
    elif target_type == sitk.Image:
        return _convert_to_sitk_image(image, accept_none=accept_none)
    elif (target_type == MedicalVolume) or (target_type == VoxelMedicalVolume):
        return _convert_to_medicalvolume(image, accept_none=accept_none)
    elif target_type == np.ndarray:
        return _convert_to_numpy_array(image, accept_none=accept_none)
    else:
        raise TypeError(f"Unsupported target type: {target_type}. Expected SimpleITK.Image, MedicalVolume, tuple, or numpy.ndarray.")


def request_sitk(_func=None, *, image_arg_names=None, convert_result=True, accept_none=True):
    """
    Decorator to convert any image type to SimpleITK.Image for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    accept_none : bool, optional
        Whether to accept None as a valid input for image arguments. If True, None will be passed through without conversion. 
        If False, a TypeError will be raised for None inputs. Default is False.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to SimpleITK.Image
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            arg_names = image_arg_names if image_arg_names is not None else [] 

            # Default to the first argument if no names are provided
            if len(arg_names) == 0:
                arg_names.append(func.__code__.co_varnames[0])

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in arg_names:
                    if convert_to is None:
                        convert_to = type(new_args[i])  # Store the original type for conversion back
                        if issubclass(convert_to, str) or issubclass(convert_to, Path):
                            convert_to = sitk.Image  # If the original type is a string (file path), convert to SimpleITK.Image
                    
                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_sitk_image(kwargs[arg_name], accept_none=accept_none)
                    else:
                        new_args[i] = _convert_to_sitk_image(new_args[i], accept_none=accept_none)

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if possible
            if convert_result:
                try:
                    return convert_to_type(result, convert_to, accept_none=True)
                except TypeError:
                    pass

            return result

        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def request_numpy(_func=None, *, image_arg_names=None, convert_result=True, accept_none=True):
    """
    Decorator to convert any image type to numpy.ndarray for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    accept_none : bool, optional
        Whether to accept None as a valid input for image arguments. If True, None will be  passed through without conversion. 
        If False, a TypeError will be raised for None inputs. Default is False.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to numpy.ndarray
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            arg_names = image_arg_names if image_arg_names is not None else []

            # Default to the first argument if no names are provided
            if len(arg_names) == 0:
                arg_names.append(func.__code__.co_varnames[0])

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in arg_names:
                    if convert_to is None:
                        convert_to = type(new_args[i])  # Store the original type for conversion back
                        if issubclass(convert_to, str) or issubclass(convert_to, Path):
                            convert_to = np.ndarray  # If the original type is a string (file path), convert to numpy.ndarray
                    
                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_numpy_array(kwargs[arg_name], accept_none=accept_none)
                    else:
                        new_args[i] = _convert_to_numpy_array(new_args[i], accept_none=accept_none)

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if convert_result:
                try:
                    return convert_to_type(result, convert_to, accept_none=True)
                except TypeError:
                    pass

            return result
            
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def request_medicalvolume(_func=None, *, image_arg_names=None, convert_result=True, accept_none=True):
    """
    Decorator to convert any image type to MedicalVolume for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    accept_none : bool, optional
        Whether to accept None as a valid input for image arguments. If True, None will be passed through without conversion. 
        If False, a TypeError will be raised for None inputs. Default is False.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to MedicalVolume
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            arg_names = image_arg_names if image_arg_names is not None else [] 

            # Default to the first argument if no names are provided
            if len(arg_names) == 0:
                arg_names.append(func.__code__.co_varnames[0])

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in arg_names:
                    if convert_to is None:
                        convert_to = type(new_args[i])  # Store the original type for conversion back
                        if issubclass(convert_to, str) or issubclass(convert_to, Path):
                            convert_to = MedicalVolume  # If the original type is a string (file path), convert to MedicalVolume

                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_medicalvolume(kwargs[arg_name], accept_none=accept_none)
                    else:
                        new_args[i] = _convert_to_medicalvolume(new_args[i], accept_none=accept_none)

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if convert_result:
                try:
                    return convert_to_type(result, convert_to, accept_none=accept_none)
                except TypeError:
                    pass

            return result
            
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def request_tuple(_func=None, *, image_arg_names=None, convert_result=True, accept_none=True):
    """
    Decorator to convert any image type to a tuple of (numpy.ndarray, metadata_array) for the decorated function.

    Parameters
    ----------
    image_arg_names : list of str
        Names of the arguments that are expected to be images.
    returns_image : bool, optional
        Whether the decorated function returns an image.
        If True, the decorator will convert the output back to the original image type passed to the function.
    accept_none : bool, optional
        Whether to accept None as a valid input for image arguments. If True, None will be passed through without conversion. 
        If False, a TypeError will be raised for None inputs. Default is False.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # Convert input images to tuple of (numpy.ndarray, metadata_array)
            new_args = list(args)
            convert_to = None
            
            # Convert to list for easier manipulation
            arg_names = list(image_arg_names) if image_arg_names is not None else []

            # Default to the first argument if no names are provided
            if len(arg_names) == 0:
                arg_names.append(func.__code__.co_varnames[0])

            for i, arg_name in enumerate(func.__code__.co_varnames):
                if arg_name in arg_names:
                    if convert_to is None:
                        convert_to = type(new_args[i])  # Store the original type for conversion back
                        if issubclass(convert_to, str) or issubclass(convert_to, Path):
                            convert_to = tuple  # If the original type is a string (file path), convert to tuple

                    if arg_name in kwargs:
                        kwargs[arg_name] = _convert_to_tuple(kwargs[arg_name], accept_none=accept_none)
                    else:
                        new_args[i] = _convert_to_tuple(new_args[i], accept_none=accept_none)

            # Call the original function
            result = func(*new_args, **kwargs)

            # Convert output back to original type if necessary
            if convert_result:
                try:
                    return convert_to_type(result, convert_to, accept_none=accept_none)
                except TypeError:
                    pass

            return result
        
        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)
    