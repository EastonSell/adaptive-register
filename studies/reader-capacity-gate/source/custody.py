"""Small self-owned-process custody primitives; no supervisor or resume support."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time


class BudgetExceeded(RuntimeError): pass


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def tree_sha(value):
    import torch
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            h.update(json.dumps(['tensor',str(x.dtype),list(x.shape)]).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'dict[')
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):add(key);add(x[key])
            h.update(b']')
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode()+b'[')
            for y in x:add(y)
            h.update(b']')
        else:h.update(json.dumps([type(x).__name__,x],sort_keys=True,allow_nan=False).encode())
    add(value)
    return h.hexdigest()


def _fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Custody:
    def __init__(self,root,*,cpu_cap=5400,wall_cap=7200,rss_cap=3*1024**3,
                 output_cap=256*1024**2,reserve=16*1024**2,start_wall=None,evidence_kind="source_fixture"):
        self.root=Path(root);self.root.mkdir(parents=False,exist_ok=False)
        self.start_cpu=0.0;self.start_wall=time.monotonic() if start_wall is None else start_wall
        self.cpu_cap=cpu_cap;self.wall_cap=wall_cap;self.rss_cap=rss_cap
        self.output_cap=output_cap;self.reserve=reserve;self._finished=False;self.index=0
        self.evidence_kind=evidence_kind;self.admission=None
        self.write_json('running.json',{'status':'running','pid':os.getpid(),'evidence_kind':evidence_kind,
            'cpu_cap':cpu_cap,'wall_cap':wall_cap,'rss_cap':rss_cap,
            'output_cap':output_cap,'metadata_reserve':reserve,
            'completion_scope':'self_owned_process_only_no_independent_supervisor'})
        self.event('run_begin',evidence_kind=evidence_kind,running_file=self.identity('running.json'))

    def identity(self,name):
        path=self.root/name
        return dict(name=name,bytes=path.stat().st_size,sha256=file_sha(path))

    def save_json(self,name,value):
        result=self.write_json(name,value)
        self.event('json_saved',**self.identity(name))
        return result

    def admit(self,admission):
        if self.admission is not None or admission.get('evidence_kind')!=self.evidence_kind:
            raise ValueError('evidence admission kind mismatch or duplicate')
        self.admission=json.loads(json.dumps(admission))
        self.save_json('admission.json',self.admission)
        self.save_json('source_manifest.json',self.admission['source_files'])
        self.save_json('plan.json',self.admission['plan'])

    def usage(self):
        return dict(cpu_seconds=time.process_time()-self.start_cpu,
                    wall_seconds=time.monotonic()-self.start_wall,
                    peak_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    output_bytes=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()))

    def check(self):
        u=self.usage()
        if u['cpu_seconds']>=self.cpu_cap or u['wall_seconds']>=self.wall_cap or u['peak_self_rss_bytes']>self.rss_cap or u['output_bytes']>self.output_cap:
            raise BudgetExceeded('self-owned capacity resource/output ceiling reached')
        return u

    def _write_new(self,name,data,*,payload=False):
        if Path(name).name!=name or not name or name.startswith('.'):
            raise ValueError('simple new output filename required')
        target=self.root/name
        if target.exists() or (self.root/(name+'.part')).exists():raise ValueError('output already exists')
        ceiling=self.output_cap-self.reserve if payload else self.output_cap
        if self.usage()['output_bytes']+len(data)>ceiling:raise BudgetExceeded('output ceiling/reserve would be exceeded')
        part=self.root/(name+'.part')
        with part.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.rename(part,target);_fsync_dir(self.root)
        return target

    def write_json(self,name,value):
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        return self._write_new(name,data)

    def event(self,event,**fields):
        self.index+=1
        row=dict(index=self.index,event=event,usage=self.usage(),**fields)
        data=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
        if self.usage()['output_bytes']+len(data)>self.output_cap:raise BudgetExceeded('event output ceiling exceeded')
        with (self.root/'events.jsonl').open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if self.index==1:_fsync_dir(self.root)
        return row

    def save_payload(self,name,payload):
        import torch
        self.check();self.event('payload_begin',name=name)
        buffer=io.BytesIO();torch.save(payload,buffer)
        target=self._write_new(name,buffer.getvalue(),payload=True)
        result=dict(name=name,bytes=target.stat().st_size,sha256=file_sha(target),content_sha256=tree_sha(payload),
                    component_sha256={str(k):tree_sha(v) for k,v in payload.items()} if isinstance(payload,dict) else None)
        self.event('payload_saved',**result);self.check()
        return result

    def finish(self,status,*,exception=None,final_check=None,**fields):
        if self._finished:raise ValueError('terminal record already exists')
        if status not in ('passed','failed','incomplete'):raise ValueError('invalid terminal status')
        if status=='passed':
            if exception is not None:raise ValueError('success cannot contain an exception')
            if final_check is not None:final_check()
            self.check()
        terminal=dict(status=status,exception=exception,evidence_kind=self.evidence_kind,usage=self.usage(),
                      accounting_boundary='sample before terminal-file write; not exact process-exit resources',
                      completion_scope='self_owned_process_only_no_independent_supervisor',**fields)
        if self.evidence_kind in ('pinned_A_capacity','random_tiny_A_capacity') and self.admission is not None:
            from .validation import validate_run,read_events
            validate_run(self.root,read_events(self.root),terminal,self.admission)
        elif status=='passed' and self.evidence_kind in ('pinned_A_capacity','random_tiny_A_capacity'):
            raise ValueError('calibration success requires frozen admission')
        # Validation is real CPU/memory work too; check its latest observed cost.
        if status=='passed':self.check()
        raw=(json.dumps(terminal,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        self.event('terminal',status=status,exception=exception,terminal_bytes=len(raw),terminal_sha256=hashlib.sha256(raw).hexdigest())
        self._write_new('terminal.json',raw);self._finished=True
        return terminal

