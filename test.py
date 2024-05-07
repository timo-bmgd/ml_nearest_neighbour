import unittest
from main import get_all_vectors, get_image

class TestImageMethods(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test fixtures
        pass

    def tearDown(self):
        # Clean up any resources used by the test methods
        pass

    def test_get_all_vectors(self):
        # Test get_all_vectors method
        test_data = [
            {b'data': [[1, 2, 3], [4, 5, 6]]},
            {b'data': [[7, 8, 9], [10, 11, 12]]}
        ]
        expected_output = [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ]
        result = get_all_vectors(test_data)
        self.assertEqual(result, expected_output)

    def test_get_image(self):
        # Test get_image method
        test_batch = {b'data': [[1, 2, 3], [4, 5, 6]]}
        image_id = 1
        expected_output = [[4, 5, 6]]
        result = get_image(test_batch, image_id)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
