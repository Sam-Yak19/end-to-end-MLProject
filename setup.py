from setuptools import find_packages,setup
from typing import List

hypen_e_dot='-e .'
def get_requirements(file_path:str)->List[str]:
    'This fnction will return the list of requirements'

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup(

    name='ML_Project',
    version='0.0.1',
    author='Samyak',
    author_email='jsamyak799@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)