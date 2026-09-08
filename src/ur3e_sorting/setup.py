from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur3e_sorting'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models/red_cylinder'), glob('models/red_cylinder/*')),
        (os.path.join('share', package_name, 'models/camera'), glob('models/camera/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anand',
    maintainer_email='anandpulikkaparambe@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = ur3e_sorting.perception_node:main',
            'sorting_node = ur3e_sorting.sorting_node:main',
            'automation_node = ur3e_sorting.automation_node:main',
            'test_moveit = ur3e_sorting.test_moveit:main',
            'yolo_detector = ur3e_sorting.yolo_detector:main',
            'locator_verification_node = ur3e_sorting.locator_verification:main',
            'pick_and_place_demo = ur3e_sorting.pick_and_place_demo:main',
            'detector_node = ur3e_sorting.detector_node:main',
            'phase1_pick_and_place = ur3e_sorting.phase1_tf_pick_and_place:main',
            'phase1_joint_pick_and_place = ur3e_sorting.phase1_joint_pick_and_place:main',
            'real_pick_and_place = ur3e_sorting.real_pick_and_place:main',
            'tf_frame_inspector = ur3e_sorting.tf_frame_inspector:main',
        ],
    },
)
