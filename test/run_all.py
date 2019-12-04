#!/usr/bin/env python

"""
Main test function to execute all tests found in the current directory
"""
import sys
import xmlrunner
from pathlib import Path

import unittest


def main():
    tests = unittest.TestLoader().discover(".", "test_*.py")
    testResult = xmlrunner.XMLTestRunner(
        output=str(Path(".").resolve() / "test-reports")
    ).run(tests)

    return not testResult.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())
