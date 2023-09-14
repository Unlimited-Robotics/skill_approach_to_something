from setuptools import setup, find_packages
import json

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('manifest.json', 'r',  encoding='utf-8') as file:
    data = json.load(file)
    SKILL_NAME = data['name']
    VERSION = data['version']
    DESCRIPTION = data.get(
        'description',
        'Raya SDK - Unlimited Robotics Software Development Kit'
    )
    AUTOR_NAME= data.get('name','Unlimited Robotics')
    AUTOR_EMAIL= data.get('email','camilo@unlimited-robotics.com')
    URL= data['repository']['url']

setup(
    name=SKILL_NAME,
    version=VERSION,
    packages=find_packages(where='skills'),
    package_dir={'': 'skills'},
    license='MIT',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTOR_NAME,
    author_email=AUTOR_EMAIL,
    url=URL,
    python_requires=">=3.8",
    download_url='',
    keywords=['skills', 'unlimited-robotics', 'gary'],
    install_requires=[]
)
