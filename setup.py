#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Laura Tiessler-Sala",
    author_email='laura.tiessler@uab.cat',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="ate heme binding cavities",
    entry_points={
        'console_scripts': [
            'hemefinder=hemefinder.__main__:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='hemefinder',
    name='hemefinder',
    packages=find_packages(include=['hemefinder', 'hemefinder.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/laura-tiessler/hemefinder',
    version='0.1.0',
    zip_safe=False,
)
