# setup.py
from setuptools import setup, find_packages

setup(
    name='snt',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',       # For data manipulation
        'openpyxl>=3.0.0',     # For reading/writing Excel files
        'numpy>=1.18.0',       # For numerical operations (e.g., pmax-like functionality)
        'xlrd>=2.0.0',         # If reading older .xls files
        'pyreadstat>=1.1.4',   # Optional: for reading Stata/SAS/SPSS if needed later
        'tabulate>=0.8.9',
        'geopandas'
    ],
    description='SNT Toolbox for data processing',
    author='Mohamed Sillah Kanu',
    author_email='sillahmohamedkanu@gmail.com',
    url='https://github.com/mohamedsillahkanu/snt-package', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
