"""
Created by: Johannes Bereiter-Payr
Created on: October 27th, 2026
This test suite was created with agentic assistance.
"""
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ormir_mids.utils.OMidsMedVolume import MedicalVolume, VoxelMedicalVolume

from ormir_xct.core.util.decorators import (
    convert_to_type,
    request_medicalvolume,
    request_numpy,
    request_sitk,
    request_tuple,
)
from ormir_xct.core.util.file_reader import file_reader


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_IMAGE_PATH = ROOT / "examples" / "images" / "test1.nii"


class TestPublicDecoratorAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = file_reader(EXAMPLE_IMAGE_PATH)

    def test_convert_to_type_supported_targets(self):
        converted_sitk = convert_to_type(self.image, sitk.Image)
        self.assertIsInstance(converted_sitk, sitk.Image)
        self.assertEqual(converted_sitk.GetSize(), self.image.A.shape)

        converted_numpy = convert_to_type(self.image, np.ndarray)
        self.assertIsInstance(converted_numpy, np.ndarray)
        self.assertEqual(converted_numpy.shape, self.image.A.shape)

        converted_tuple = convert_to_type(self.image, tuple)
        self.assertIsInstance(converted_tuple, tuple)
        self.assertEqual(len(converted_tuple), 2)
        self.assertIsInstance(converted_tuple[0], np.ndarray)
        self.assertEqual(converted_tuple[0].shape, self.image.A.shape)

        converted_medical = convert_to_type(self.image, MedicalVolume)
        self.assertTrue(hasattr(converted_medical, "A"))
        self.assertEqual(converted_medical.A.shape, self.image.A.shape)

    def test_convert_to_type_numpy_to_medicalvolume(self):
        array = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
        converted = convert_to_type(array, MedicalVolume)
        self.assertTrue(hasattr(converted, "A"))
        self.assertEqual(converted.A.shape, array.shape)
        np.testing.assert_array_equal(converted.A, array)

    def test_convert_to_type_rejects_unsupported_target(self):
        with self.assertRaises(TypeError):
            convert_to_type(self.image, int)

    def test_request_sitk_round_trip(self):
        @request_sitk
        def decorated(image):
            self.assertIsInstance(image, sitk.Image)
            return image

        result = decorated(self.image)
        self.assertIsInstance(result, type(self.image))
        self.assertEqual(result.A.shape, self.image.A.shape)

    def test_request_numpy_round_trip(self):
        @request_numpy
        def decorated(image):
            self.assertIsInstance(image, np.ndarray)
            return image

        result = decorated(self.image)
        self.assertIsInstance(result, type(self.image))
        self.assertEqual(result.A.shape, self.image.A.shape)

    def test_request_medicalvolume_round_trip(self):
        @request_medicalvolume
        def decorated(image):
            self.assertIsInstance(image, (MedicalVolume, VoxelMedicalVolume))
            return image

        result = decorated(convert_to_type(self.image, sitk.Image))
        self.assertIsInstance(result, sitk.Image)
        self.assertEqual(result.GetSize(), convert_to_type(self.image, sitk.Image).GetSize())

    def test_request_sitk_image_arg_names_for_non_image_first_parameter(self):
        @request_sitk(image_arg_names=("image_b",))
        def decorated(other, image_b):
            self.assertIsInstance(image_b, sitk.Image)
            self.assertEqual(other, 123)
            return image_b

        result = decorated(123, self.image)
        self.assertIsInstance(result, type(self.image))
        self.assertEqual(result.A.shape, self.image.A.shape)

    def test_request_tuple_default_argument_selection(self):
        @request_tuple
        def decorated(image):
            self.assertIsInstance(image, tuple)
            self.assertIsInstance(image[0], np.ndarray)
            return image

        result = decorated(self.image)
        self.assertIsInstance(result, type(self.image))
        self.assertEqual(result.A.shape, self.image.A.shape)

    def test_request_decorators_respect_accept_none(self):
        @request_sitk(accept_none=True)
        def accept_none(image):
            return image

        @request_sitk(accept_none=False)
        def reject_none(image):
            return image

        self.assertIsNone(accept_none(None))
        with self.assertRaises(TypeError):
            reject_none(None)


if __name__ == "__main__":
    unittest.main()
