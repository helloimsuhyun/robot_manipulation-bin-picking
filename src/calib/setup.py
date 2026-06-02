from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'calib'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # launch 파일 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # handeye config json 설치
        (
            os.path.join('share', package_name, 'config', 'handeye_capture_rs'),
            glob('config/handeye_capture_rs/*.json')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choisuhyun',
    maintainer_email='chsuk02@hanyang.ac.kr',
    description='Calibration and object pose transform package',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'object_pose_transform_node = calib.object_pose_transform_node:main',
            'sixd_pose_transform_node = calib.sixd_pose_transform_node:main',
        ],
    },
)
