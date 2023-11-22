import setuptools

setuptools.setup(
    name='rai-chem',
    version='0.0.4',
    url='https://github.com/vtarasv/rai-chem.git',
    download_url='https://github.com/vtarasv/rai-chem/archive/refs/tags/0.0.4.tar.gz',
    install_requires=['tqdm', 'numpy', 'pandas', 'scipy', 'rdkit-pypi==2022.9.1'],
    include_package_data=True,
    packages=setuptools.find_packages(),
    license='MIT',
    author='vtarasv',
    author_email='taras.voitsitskyi@receptor.ai',
    description='',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
      ],
)
