import os
from setuptools import setup, Extension
from torch.utils import cpp_extension


curdir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(curdir, 'BitmapTensor.cpp')
]
setup(name='edgify_tensor',
      ext_modules=[cpp_extension.CppExtension('edgify_tensor', sources, extra_compile_args=['-I/usr/local/include'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

