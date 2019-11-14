from setuptools import setup, find_packages

setup(
    name='alpacka',
    description='AwareLab PACKAge - internal RL framework',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['gym', 'numpy', 'pytest']
)
