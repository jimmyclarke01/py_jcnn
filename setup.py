from setuptools import setup

setup(
   name='jcnn',
   version='1.0',
   description='Neural Network Library',
   author='James Clarke',
   author_email='james.clarke42@gmail.com',
   packages=['jcnn'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)