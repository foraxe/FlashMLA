import os
from pathlib import Path
import subprocess
import sys
root=Path('/workspace/vllm_dsv41')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',VLLM_USE_V2_MODEL_RUNNER='1',VLLM_DEEP_GEMM_WARMUP='skip',PYTHONPATH=f'{root}/artifacts/fused-output:{root}/runtime-deps/nvidia_cutlass_dsl/dsl_packages:{root}/runtime-deps:{root}/vllm-fused-out')
for label,variant in [('b1','candidate'),('b2','candidate'),('a2','baseline')]:
    path=root/'artifacts/fused-output'/f'mrv2-{label}'
    print(f'START {label} {variant}',flush=True)
    with path.with_suffix('.log').open('w') as log:
        subprocess.run([str(root/'vllm-fused-out/.venv/bin/python'),str(root/'artifacts/fused-output/fused_output_bench_mrv2.py'),'--variant',variant,'--contexts','17','--trials','6','--output',str(path.with_suffix('.json'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'DONE {label}',flush=True)
