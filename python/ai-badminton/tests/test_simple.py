# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

from ai_badminton import court, hit_detector, pose, trajectory

class TestSimple(unittest.TestCase):
    def test_nothing(self):
        self.assertEqual(0, 0)

if __name__ == '__main__':
    unittest.main()
