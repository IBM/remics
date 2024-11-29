from setuptools import setup, find_packages

setup(
    name='remics',  # Your project name
    version='0.1.3',    # Version of your project
    description='A Redescription-based framework for multi-omics analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aritra Bose',
    author_email='a.bose@ibm.com',
    url='https://github.com/IBM/remics/',  # Project URL
    packages=find_packages(),  # Automatically find packages in your project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',  # Or your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
