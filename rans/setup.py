from setuptools import setup, Extension
from Cython.Build import cythonize


setup(
    name='ranstools',
    ext_modules=cythonize([
        Extension(
            "rans",
            sources=["rans.pyx"]
        )
    ]),
    zip_safe=False
)
