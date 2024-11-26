from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['labrob_crowd_navigation_utils'],
    package_dir={'': 'src'}
)
setup(**d)