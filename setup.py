from setuptools import setup

setup(name='dynalearn',
      version='0.1',
      description='Boltzmann machine framework for dynamics on network learning.',
      url='https://github.com/charlesmurphy1/dynalearn',
      author='Charles Murphy',
      author_email='charles.murphy121@gmail.com',
      license='MIT',
      packages=['boltzmann_machine', 'dynamics', 'history', 'trainer', 'utilities'],
      zip_safe=False)

#requirement