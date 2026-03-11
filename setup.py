from setuptools import find_packages,setup;
from typing import List;

HYPEN_E_DOT='-e .'
def get_requirements(file:str)->List[str]:
    #### This function will return the list of requirements #####
    req=[];
    with open(file,'r') as fileobj:
        req=fileobj.readlines()
    req=[pckg.replace('\n','') for pckg in req]
    if HYPEN_E_DOT in req:
        req.remove(HYPEN_E_DOT);
    return req;


setup(
    name='mlproject',
    version='1.0.0',
    author='Tamal',
    author_email='tamalkoley121@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)