import setuptools

setuptools.setup(
    name="pyflex_utils",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="TODO",
    install_requires=[
        "numpy",
        "trimesh",
        "click"
    ],  # do not add pyflex here, has to be loaded manually
    packages=["pyflex_utils"],
)
