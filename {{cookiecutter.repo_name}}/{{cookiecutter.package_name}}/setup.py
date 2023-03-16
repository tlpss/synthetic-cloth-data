import setuptools
from setuptools import find_packages

setuptools.setup(
    name="{{cookiecutter.package_name}}",
    version="0.0.1",
    author="{{cookiecutter.author_name}}",
    author_email="{{cookiecutter.author_email}}",
    description="TODO",
    install_requires=[
        "numpy",
    ],
    packages=find_packages(),
)
