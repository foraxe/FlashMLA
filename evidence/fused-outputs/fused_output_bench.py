import argparse
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


class BenchWorker(Worker):
    def load_model(self, **kwargs):
        import torch
        from vllm.models.deepseek_v4_1.nvidia.flashmla_fused import DeepseekV4FlashMLAFusedAttention as Fused
        if os.environ['FUSED_OUTPUT_VARIANT'] == 'baseline':
            from fused_baseline import DeepseekV4FlashMLAFusedAttention as Baseline
            for name in ('_alloc_attn_out','_finish_o_proj','forward_mqa',
                         '_forward_decode_fused','_forward_decode_split_kv','_forward_prefill_fused'):
                setattr(Fused,name,getattr(Baseline,name))
        original=Fused._wo_a_einsum
        self._projection_original=original
        def count_projection(layer,*args,**kw):
            counts=layer.__dict__.setdefault('_projection_counts', {'capture':0,'eager':0})
            key='capture' if torch.cuda.is_current_stream_capturing() else 'eager'
            counts[key]+=1
            return original(layer,*args,**kw)
        Fused._wo_a_einsum=count_projection
        super().load_model(**kwargs)
        self._replays={}
        wrapper=self.model_runner.model
        self._replay_original=wrapper._replay
        def count_replay(entry,*args,**kw):
            key=str(entry.batch_descriptor)
            self._replays[key]=self._replays.get(key,0)+1
            return self._replay_original(entry,*args,**kw)
        wrapper._replay=count_replay
        self._steps={'prefill':0,'decode':0,'idle':0,'tokens':[]}
        original_execute=self.model_runner.execute_model
        def count_step(scheduler_output,*args,**kw):
            lens=list(scheduler_output.num_scheduled_tokens.values())
            key = 'idle' if not sum(lens) else ('prefill' if max(lens)>1 else 'decode')
            self._steps[key]+=1
            self._steps['tokens'].append(sum(lens))
            return original_execute(scheduler_output,*args,**kw)
        self.model_runner.execute_model=count_step

    def probe(self,action):
        import copy
        from vllm.models.deepseek_v4_1.nvidia.flashmla_fused import DeepseekV4FlashMLAFusedAttention as Fused
        model=self.model_runner.get_model()
        if action in ('reset','timing'):
            self._replays.clear()
            self._steps={'prefill':0,'decode':0,'idle':0,'tokens':[]}
            for layer in model.modules():
                if isinstance(layer,Fused):layer._projection_counts={'capture':0,'eager':0}
        if action=='timing':
            Fused._wo_a_einsum=self._projection_original
            self.model_runner.model._replay=self._replay_original
        counts={'capture':0,'eager':0}
        for layer in model.modules():
            if isinstance(layer,Fused):
                for k,v in getattr(layer,'_projection_counts',{}).items():counts[k]+=v
        return {'rank':self.rank,'projection_calls':counts,'replays':dict(self._replays),'steps':copy.deepcopy(self._steps)}


def main(args):
    import math
    from vllm import LLM,SamplingParams
    from vllm.tokenizers import get_tokenizer
    model='/data/models/DeepSeek-V4.1-Flash'
    llm=LLM(model=model,tokenizer_mode='deepseek_v41',tensor_parallel_size=4,
        language_model_only=True,max_model_len=40960,max_num_seqs=4,max_num_batched_tokens=8192,
        kv_cache_memory_bytes=2**30,enable_prefix_caching=False,seed=0,disable_log_stats=False,
        attention_config={'dsv4_fused_attention':True,'dsv4_fused_decode_min_tokens':args.decode_min_tokens},
        compilation_config={'cudagraph_capture_sizes':[1,2,4,32,64]},
        kernel_config={'enable_jit_warmup':False,'enable_cutedsl_warmup':False,'enable_flashinfer_autotune':False},
        worker_cls='fused_output_bench.BenchWorker')
    llm.collective_rpc('probe',args=('reset',))
    check=llm.chat([{'role':'user','content':'What is 17 times 19? Return only the integer.'}],
        SamplingParams(temperature=0,max_tokens=8),chat_template_kwargs={'thinking':False})[0]
    diagnostic=llm.collective_rpc('probe',args=('read',))
    assert check.outputs[0].text.strip()=='323',check.outputs[0].text
    eager_expected=40 if args.variant=='baseline' else 0
    assert all(x['projection_calls']['eager']==eager_expected for x in diagnostic),diagnostic
    llm.collective_rpc('probe',args=('timing',))
    tok=get_tokenizer(model,tokenizer_mode='deepseek_v41')
    short=list(check.prompt_token_ids)
    seed_ids=tok.encode('The sky is blue. ',add_special_tokens=False)
    prompts={17:short}
    assert len(short)==17
    for n in (8192,32768):prompts[n]=(seed_ids * math.ceil(n/len(seed_ids)))[:n]
    mixed=llm.generate([{'prompt_token_ids':short},{'prompt_token_ids':prompts[32768][:16384]}],
        SamplingParams(temperature=0,max_tokens=16,ignore_eos=True),use_tqdm=False)
    assert all(r.metrics is not None and not r.metrics.is_corrupted for r in mixed)
    mixed_outputs=[r.outputs[0].token_ids for r in mixed]
    results=[]
    path=Path(args.output)
    def save():path.write_text(json.dumps({'variant':args.variant,'diagnostic':diagnostic,'mixed_outputs':mixed_outputs,'results':results},indent=2))
    save()
    for n in args.contexts:
        for trial in range(args.trials+2):
            llm.collective_rpc('probe',args=('reset',))
            result=llm.generate({'prompt_token_ids':prompts[n]},SamplingParams(temperature=0,max_tokens=64,ignore_eos=True),use_tqdm=False)[0]
            stats=llm.collective_rpc('probe',args=('read',))
            assert all(x['steps']['prefill']==math.ceil(n/8192) and x['steps']['decode']==63 for x in stats),stats
            metrics=result.metrics
            assert metrics is not None and not metrics.is_corrupted and metrics.num_preemptions==0
            assert metrics.first_token_ts>metrics.scheduled_ts>0 and metrics.last_token_ts>metrics.first_token_ts
            row={'context':n,'trial':trial,'warmup':trial<2,'prefill_ms':(metrics.first_token_ts-metrics.scheduled_ts)*1000,
                 'ttft_ms':metrics.first_token_latency*1000,'tpot_ms':(metrics.last_token_ts-metrics.first_token_ts)/63*1000,
                 'token_ids':result.outputs[0].token_ids,'steps':stats[0]['steps']}
            assert len(row['token_ids'])==64
            results.append(row);save()
            print(json.dumps({k:v for k,v in row.items() if k not in ('token_ids','steps')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['baseline','candidate'],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--contexts',nargs='+',type=int,default=[17,8192,32768]);parser.add_argument('--trials',type=int,default=6);parser.add_argument('--decode-min-tokens',type=int,default=1)
    args=parser.parse_args();os.environ['FUSED_OUTPUT_VARIANT']=args.variant;main(args)
