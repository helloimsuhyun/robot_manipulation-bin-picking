from setuptools import setup
from glob import glob
import os

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chu',
    maintainer_email='chu@example.com',
    description='RB5 peg-in-hole controller using rbpodo and ROS 2 parameters.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'peg_in_hole_controller = control.main:main',
            'keyboard_continue_node = control.keyboard_continue_node:main',
        ],
    },
)
