import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pytorch_helper',
    version='0.5.0',
    author='Zhirui Dai',
    author_email='daizhirui@hotmail.com',
    description='A package that provides a useful helper framework for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/daizhirui/pytorch-helper',
    project_urls={
        'Bug Tracker': 'https://github.com/daizhirui/pytorch-helper/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.8',
)
