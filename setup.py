from setuptools import find_packages, setup

def get_requirements(file_name):
    requirements = []
    with open(file_name, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
        return requirements

setup(
    name='ImageCaptionGenerator',
    version='0.0.1',
    author='Bhavana',
    author_email='abc@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('./requirements.txt')

)