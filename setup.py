import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='gls',
    version='0.4.9',
    author='552 Group 4',
    author_email='aidan.williams@mail.mcgill.ca',
    description='GLS Layer Implementation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/BenMacnaughton/chaos_inspired_deepnet',
    license='MIT',
    packages=['gls'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'torch',
        'torchvision'
    ],
)