from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='tt_nn_quant',
      ext_modules=[
            cpp_extension.CppExtension('tt_nn_quant', 
                  ['torch_wrap.cpp', 'tt_top.cpp', 'tt_contract.cpp', 'random.cpp'],
                  extra_compile_args=['-mavx2', '-g', '-D QUANTIZE']),
            cpp_extension.CppExtension('tt_nn', 
                  ['torch_wrap.cpp', 'tt_top.cpp', 'tt_contract.cpp', 'random.cpp'],
                  extra_compile_args=['-mavx2', '-g'])
      ],

      cmdclass={'build_ext': cpp_extension.BuildExtension},
      extra_compile_args=['-mavx2'])

# setup(name='tt_nn',
      # ext_modules=[cpp_extension.CppExtension('tt_nn', 
      #       ['torch_wrap.cpp', 'tt_top.cpp', 'tt_contract.cpp', 'random.cpp'],
      #       extra_compile_args=['-mavx2', '-g'])],
      # cmdclass={'build_ext': cpp_extension.BuildExtension},
      # extra_compile_args=['-mavx2'])

