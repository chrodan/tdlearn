from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "tdbench",
    author = "Christoph Dann",
    author_email="cdann@cdann.de",
    version = "git",
    ext_modules = cythonize('swingup_ode.pyx'),
)
