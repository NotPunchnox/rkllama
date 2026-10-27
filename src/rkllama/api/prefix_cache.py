"""Opt-in, single-slot native prefix cache. Full token input is always submitted.

The cache belongs to one worker and one model handle. It is never selected by
message count or user-supplied filenames. Native state is released/full-cleared
before every request; incomplete prefixes, changed policy and branches cannot
reuse a mismatching snapshot.
"""
import tempfile
from pathlib import Path

class PrefixCache:
    def __init__(self,model,parent,min_tokens=256,max_bytes=1024**3):
        self.model=model;self.parent=Path(parent);self.min_tokens=min_tokens;self.max_bytes=max_bytes
        self.root=None;self.path=None;self.prefix=();self.loaded=False;self.disabled=False;self.tainted=False
        self.cancelled=lambda:False
        self.last_hit=False;self.last_priming_metrics=[]

    def _release_clear(self):
        if self.loaded:
            if self.model.release_prompt_cache(self.model.handle)!=0:
                self.tainted=True;raise RuntimeError('Native prompt-cache release failed')
            self.loaded=False
        if self.model.rkllm_clear_kv_cache(self.model.handle,0,None,None)!=0:
            self.tainted=True;raise RuntimeError('Native full-cache clearing failed')

    def invalidate(self):
        self._release_clear();self.prefix=()
        if self.path and self.path.exists():self.path.unlink()

    def invoke(self,mode,kind,inputs,options):
        from . import callback
        ids,legacy=inputs;ids=list(ids)
        assert legacy is None
        self.last_hit=False;self.last_priming_metrics=[]
        self._release_clear()
        try:
            if self.cancelled():return -1
            if not self.disabled and len(ids)>=self.min_tokens:
                hit=bool(self.prefix) and len(self.prefix)<len(ids) and tuple(ids[:len(self.prefix)])==self.prefix
                self.last_hit=hit
                if not hit:
                    self.prefix=()
                    if self.path and self.path.exists():self.path.unlink()
                    if self.root is None:
                        self.parent.mkdir(parents=True,exist_ok=True)
                        self.root=tempfile.TemporaryDirectory(prefix='rkllama-prefix-',dir=self.parent)
                        self.path=Path(self.root.name)/'prefix.bin'
                    prefix=ids[:-32]
                    callback.suppress_output=True
                    self.model._prefix_save_path=str(self.path)
                    try:
                        status=self.model._run(mode,kind,(prefix,None),{**(options or {}),'max_new_tokens':1})
                        self.last_priming_metrics=list(callback.global_metrics)
                        if status!=0 or len(self.last_priming_metrics)!=4:
                            raise RuntimeError('Native prefix priming failed')
                    finally:
                        self.model._prefix_save_path=None;callback.suppress_output=False;callback.reset_callback_state()
                    if self.cancelled():self.invalidate();return -1
                    if self.path.is_file() and 0<self.path.stat().st_size<=self.max_bytes:
                        self.prefix=tuple(prefix)
                    else:
                        if self.path.exists():self.path.unlink()
                        self.disabled=True
                    self._release_clear()
                if self.cancelled():self.invalidate();return -1
                if self.prefix:
                    if self.model.rkllm_load_prompt_cache(self.model.handle,str(self.path).encode())!=0:
                        self.tainted=True;raise RuntimeError('Native prompt-cache load failed')
                    self.loaded=True
            if self.cancelled():self.invalidate();return -1
            status=self.model._run(mode,kind,(ids,None),options)
            if status!=0:self.invalidate()
            return status
        except BaseException:
            callback.suppress_output=False
            self.invalidate()
            raise

    def close(self):
        try:self._release_clear()
        finally:
            self.prefix=()
            if self.root:self.root.cleanup();self.root=None;self.path=None
