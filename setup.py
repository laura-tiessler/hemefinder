#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
readme = (this_directory / "README.rst").read_text()

requirements = [
    'numpy',
    'psutil',
    'pyKVFinder',
    'scikit-learn',
    'typer'
]

test_requirements = requirements + ['pytest']

setup(
    author="Laura Tiessler-Sala",
    author_email='laura.tiessler@uab.cat',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="Prediction of heme binding cavities",
    entry_points={
        'console_scripts': [
            'hemefinder=hemefinder.__main__:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='hemefinder',
    long_description_content_type='text/x-rst',
    name='hemefinder',
    packages=find_packages(include=['hemefinder', 'hemefinder.*']),
    package_data={'': ['*.json', '*.pdb']},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/laura-tiessler/hemefinder',
    version='0.0.1',
    zip_safe=False,
)
