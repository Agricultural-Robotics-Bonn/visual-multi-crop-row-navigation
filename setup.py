from glob import glob
import os

from setuptools import setup

package_name = 'visual_multi_crop_row_navigation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alireza',
    maintainer_email='alireza.ahmadi@uni-bonn.todo',
    description='phenobot multicrop row visual servoing navigation',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vs_navigation = visual_multi_crop_row_navigation.main:main'
        ],
    }
)
