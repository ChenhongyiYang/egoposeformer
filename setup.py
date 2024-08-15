#!/usr/bin/env python

from setuptools import find_packages, setup

print(f"Installing {find_packages()}")
setup(
    name="pose_estimation",
    version="0.0.1",
    description="EgoPoseFormer: A Simple Baseline for Egocentric 3D Human Pose Estimation",
    author="Chenhongyi Yang",
    author_email="chenhongyi.yang@ed.ac.uk",
    packages=find_packages(),
)