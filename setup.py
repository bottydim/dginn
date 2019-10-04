from setuptools import setup

from setuptools import setup, find_packages

description = 'Library for Interpretable Deep Learning'
long_description = description
version = '0.1.0.dev1'

setup(name='dginn',
      version=version,
      description=description,
      long_description=long_description,
      # url='https://github.com/bottydim/xai',
      authors=['Botty Dimanov', "Dmitry Khazdan", "Plamen Mangov"],
      author_email='develop@uvhotspot.com',
      license='Apache License 2.0',
      packages=find_packages(exclude=['*.debug', 'debug.*', '*.debug.*', 'contrib', 'docs', 'tests*', 'test*']),
      # packages = ["xai"],
      zip_safe=False,
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 2 - Pre-Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Build Tools',
          'Topic :: System :: Monitoring'

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: Apache Software License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          # 'Programming Language :: Python :: 2',
          # 'Programming Language :: Python :: 2.6',
          # 'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          # 'Programming Language :: Python :: 3.2',
          # 'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.5',
          ],
      keywords='interpretable deep learning',
      # install_requires=['keras','tensorflow-gpu','pandas','seaborn',''],
      python_requires='~=3.5'
      )


'''
VERSIONING:

1.2.0.dev1  # Development release
1.2.0a1     # Alpha Release
1.2.0b1     # Beta Release
1.2.0rc1    # Release Candidate
1.2.0       # Final Release
1.2.0.post1 # Post Release
15.10       # Date based release
23          # Serial release




    MAJOR version when they make incompatible API changes,

    MINOR version when they add functionality in a backwards-compatible manner, and

    MAINTENANCE version when they make backwards-compatible bug fixes.

'''