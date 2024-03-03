from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().split('\n')

setup(
    name = 'megaDNA',
    packages = find_packages(),
    version = '1.0',
    license =  'MIT',
    description = 'MegaDNA: a long-context generative model of bacteriophage genome',
    author = 'Bin Shao, Jiawei Yan',
    url = 'https://github.com/lingxusb/megaDNA',
    author_email = 'shaobinlx@gmail.com',
    install_requires = requirements,
    python_requires = '>=3.8',
)
