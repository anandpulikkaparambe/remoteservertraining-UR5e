from setuptools import find_packages, setup

package_name = 'ur3e_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'sensor_msgs', 'geometry_msgs', 'tf2_ros', 'numpy', 'gymnasium', 'stable_baselines3'],
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
            'train_ppo = ur3e_rl.train_sb3:main',
            'train_sac = ur3e_rl.train_sac:main',
            'test_policy = ur3e_rl.test_policy:main',
            'real_phase2_dry_run = ur3e_rl.real_phase2_dry_run:main',
            'real_phase2_closed_loop_dry_run = ur3e_rl.real_phase2_closed_loop_dry_run:main',
            'real_phase2_unclamped_run = ur3e_rl.real_phase2_unclamped_run:main',
            'eval_legacy_grasp_checkpoint = ur3e_rl.eval_legacy_grasp_checkpoint:main',
            'handoff_passive_watcher = ur3e_rl.handoff_passive_watcher:main',
        ],
    },
)
