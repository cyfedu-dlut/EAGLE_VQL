"""
Setup script for EAGLE package.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='eagle',
    version='1.0.0',
    description='EAGLE',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='YFCao',
    author_email='yfcao@mail.dlut.edu.cn',
    url='',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tools', 'configs']),
    install_requires=read_requirements(),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='computer-vision deep-learning segmentation tracking visual-queries egocentric',
    entry_points={
        'console_scripts': [
            'eagle-train=tools.train:main',
            'eagle-test=tools.test:main',
            'eagle-inference=tools.inference:main',
        ],
    },
)