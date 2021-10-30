import setuptools

from src.pytorch_helper import __version__

requires = [
    'nvidia-ml-py',
    'numpy',
    'matplotlib',
    'pillow',
    'opencv-python',
    'ruamel.yaml',
    'tqdm',
    'colorama',
    'colorlog',
    # 'psutil',
    # 'PyPDF4',
    'tensorboard',
    'tensorboardX',
    'torch_tb_profiler',
    'snakeviz',
    'six>=1.7',
    'blessed>=1.17.1',
    'setuptools',
    'wheel',
]

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pytorch_helper',
    version=__version__,
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
    install_requires=requires,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.8',
)
