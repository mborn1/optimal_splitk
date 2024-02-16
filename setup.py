import setuptools
import site
import re

site.ENABLE_USER_SITE = 1

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('src/optimal_splitk/__init__.py') as f:
    version = re.search(r'__version__ = [\'"](.*)[\'"]', f.read()).group(1)

setuptools.setup(
    name="optimal_splitk",
    version=version,
    author="Mathias Born",
    author_email="mathias.born@kuleuven.be",
    license_files = ('LICENSE',),
    description="DOE library for optimal split^k-plot designs with the coordinate exchange algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        'numba==0.55.1',
        'numpy==1.21.5',
        'tqdm==4.64.0',
        'scipy==1.8.0',
    ],
    extras_require={
        'dev': [
            'sphinx~=4.4',
            'docutils<0.18',
            'numpydoc~=1.2',
            'pydata_sphinx_theme~=0.7',
            'sphinx-copybutton~=0.5'
        ],
        'examples': [
            'pandas==1.5.0',
            'openpyxl==3.0.10'
        ]
    }
)
