from setuptools import setup


# packages = ["dynalearn",
#             "dynalearn.dynamics",
#             "dynalearn.model",
#             "dynalearn.trainer",
#             "dynalearn.history",
#             "dynalearn.utilities"]

packages = ["dynalearn",
            "dynalearn.dynamics",
            "dynalearn.model",
            "dynalearn.trainer",
            "dynalearn.utilities"]


extras_require = {'all': ['numpy', 'matplotlib', 'torch', 'networkx']}


setup(name='dynalearn',
      version='0.1',
      description='Boltzmann machine framework for dynamics on network learning.',
      url='https://github.com/charlesmurphy1/dynalearn',
      author='Charles Murphy',
      author_email='charles.murphy121@gmail.com',
      license='MIT',
      packages=packages,
      requirement= extras_require,
      zip_safe=False)

#requirement