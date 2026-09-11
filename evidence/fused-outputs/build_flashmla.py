import re
from pathlib import Path
from torch.utils.cpp_extension import load

root=Path('/workspace/vllm_dsv41')
source=root/'flashmla-out'
cmake=(root/'vllm-fused-out/cmake/external_projects/flashmla.cmake').read_text()
files=list(dict.fromkeys(re.findall(r'\$\{flashmla_SOURCE_DIR\}/([^\s)]+\.(?:cpp|cu))',cmake)))
files=[f for f in files if not f.startswith('csrc/extension/')]
assert files and all((source/f).is_file() for f in files)
build=root/'flashmla-build';build.mkdir(exist_ok=True)
units=build/'units';units.mkdir(exist_ok=True)
unique_sources=[]
for i,name in enumerate(files):
    unit=units/f'{i:03d}_{Path(name).name}'
    text=f'#include "{source/name}"\n'
    if not unit.exists() or unit.read_text()!=text:
        unit.write_text(text)
    unique_sources.append(str(unit))
flags=['-O3','-std=c++20','-DTORCH_TARGET_VERSION=0x020a000000000000','-DUSE_CUDA','-DPy_LIMITED_API=0x03090000']
module=load(name='_flashmla_C', sources=unique_sources,
 extra_cflags=flags,
 extra_cuda_cflags=flags+['--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math',
  '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__',
  '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--threads=2','-gencode=arch=compute_100a,code=sm_100a'],
 extra_include_paths=[str(source/'csrc'),str(source/'csrc/kerutils/include'),str(source/'csrc/cutlass/include'),str(source/'csrc/cutlass/tools/util/include'),'/usr/local/cuda/include/cccl'],
 build_directory=str(build),with_cuda=True,verbose=True)
print(module.__file__)
