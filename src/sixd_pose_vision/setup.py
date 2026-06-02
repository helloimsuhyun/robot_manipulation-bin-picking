from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'sixd_pose_vision'


def package_files(directory):
    paths = []
    if not os.path.isdir(directory):
        return paths

    for root, _, files in os.walk(directory):
        for filename in files:
            paths.append(os.path.join(root, filename))

    return paths


def data_file_entries(directory):
    entries = []

    for filepath in package_files(directory):
        install_dir = os.path.join(
            'share',
            package_name,
            os.path.dirname(filepath),
        )
        entries.append((install_dir, [filepath]))

    return entries


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            'share/' + package_name + '/launch',
            glob('launch/*.launch.py'),
        ),

        *data_file_entries('weights'),
        *data_file_entries('CAD'),
        *data_file_entries('templates'),
        *data_file_entries('config'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choisuhyun',
    maintainer_email='chsuk02@hanyang.ac.kr',
    description='6D pose vision pipeline using YOLO segmentation, FoundationPose, and template-based insert pose estimation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mixed_pose_vision_node = sixd_pose_vision.mixed_pose_vision_node:main',
            'keyboard_object_trigger_node = sixd_pose_vision.keyboard_object_trigger_node:main',
        ],
    },
)
