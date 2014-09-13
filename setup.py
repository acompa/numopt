#! /usr/bin/env python

"""
Set things up!
"""

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(name='numopt',
      version='1.0',
      description='Numerical optimization methods.',
      author='Alex Companioni',
      author_email='achompas@gmail.com',
      url='http://acompa.net',
      package_data={"config": ["requirements*.txt"]},
      cmdclass={"test": PyTest},
      install_requires=open("requirements.txt", 'r').read(),
      packages=find_packages())
