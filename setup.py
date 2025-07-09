from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
    name='uni',
    version='0.1.0',
    description='HistoTME',
    url='https://github.com/spatkar/HistoTME',
    author='SP, AC',
    author_email='',
    license='CC BY-NC 4.0',
    packages=find_packages(exclude=['__dep__', 'assets']),
    install_requires=["torch>=2.0.1", "torchvision", "timm==0.9.16", 
                      "numpy<2", "pandas", "scikit-learn", "tqdm",
                      "transformers","opencv-python","openslide-python",
                      "openslide-bin"],

    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: CC BY-NC 4.0",
]
)
