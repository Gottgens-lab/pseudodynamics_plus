"""
PINN_dyanmics: Physics informed neural network for solving pseudodynamics PDE
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Set __version__ for the project.
# exec(open("./inDecay/version.py").read())

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    

setup(
    name='pseudodynamics',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.3',

    description='Pseudodynamics+: Physics informed neural network for solving pseudodynamics PDE',
    long_description=long_description,

    # The project's main homepage.
    url='https://gitlab.developers.cam.ac.uk/jcbc/csci/gottgenslab/pseudodynamics_plus',

    # Author details
    author=['Weizhong Zheng'],
    author_email='wz369@cam.ac.uk',

    # Choose your license
    license='Apache-2.0',

    # What does your project relate to?
    keywords=['deep learning', 'pytorch', 
              'single-cell', 'time-serires data'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    # entry_points={
    #       'console_scripts': [
    #           ],
    #       }, 

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    
    install_requires = ['numpy>=1.9.0', 'scipy>=1.4.0', 'matplotlib', 'statsmodels>=0.13', 'seaborn', 'torchcfm', 'scanpy', 'palantir'],
    include_package_data=True,


    extras_require={
        'docs':  [
            'myst-nb',
            'sphinx',
            'sphinx-autodoc-typehints',
            'sphinx-book-theme>=1.1.4',
            'sphinx-copybutton>=0.5.2',
            'sphinx-tabs==3.4.7',
            'sphinxcontrib-bibtex==2.6.5',
            'sphinxcontrib-katex>=0.9.11',
            'sphinxext-opengraph>=0.13.0',
            "furo >=2022.09.29",
            "scanpydoc >=0.5",
            "ipython >=7.1.1",
            "ipykernel",
            "sphinx_design >=0.3.0",
            "sphinx_autodoc_typehints >=1.10.3",
        ]},

)
