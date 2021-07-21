import setuptools


with open("requirements.txt", "r") as fp:
    required = fp.read().splitlines()

setuptools.setup(
    name="EMM",     # Add a project name
    version="0.1.0",
    author="Lucas Jevtic",
    author_email="lucas.jevtic@maths.ox.ac.uk",    # Add an email
    description="EMM: Entropic marginal minimisation. Used to generate semi-synthetic"+
                        " datasets from marginal statistics.",     # Add a description
    url="https://github.com/LucasKIJ/EMM.git",     # Add a github url
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=required,
    extras_require={
        "test": ["pytest"]
    }
)