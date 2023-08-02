from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    Takes in file_path, a path to a requirements file
    Returns a list of requirements
    '''
    with open(file_path, 'r') as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements 

setup(
    name='project',
    version='0.0.1',
    author='Akash',
    author_email='akashviyer@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)