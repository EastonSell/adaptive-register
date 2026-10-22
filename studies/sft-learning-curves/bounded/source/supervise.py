"""Sampled Linux process supervision, including subprocess termination on budget breach."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
from evidence import reserve, atomic, append, digest, verify_freeze
ROOT=Path(__file__).resolve().parent


def parse_sample(stat_text,status_text):
    # Process utime/stime aggregate its threads; no children are spawned by runner.
    stat=stat_text.rsplit(')',1)[1].split()
    ticks=int(stat[11])+int(stat[12])
    rss_lines=[line.split() for line in status_text.splitlines() if line.startswith('VmRSS:')]
    if len(rss_lines)!=1 or len(rss_lines[0])!=3 or rss_lines[0][2]!='kB':
        raise ValueError('Missing, duplicate or malformed VmRSS accounting')
    rss_kib=int(rss_lines[0][1])
    if ticks<0 or rss_kib<=0: raise ValueError('Invalid CPU/RSS accounting')
    return ticks/os.sysconf('SC_CLK_TCK'),rss_kib*1024


def sample(pid):
    return parse_sample(Path(f'/proc/{pid}/stat').read_text(),Path(f'/proc/{pid}/status').read_text())


def kill_live(child):
    if child.poll() is not None: return
    try: os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:
        if child.poll() is None: raise


def monitor(command,out,cfg,env=None):
    start=time.monotonic(); peak=0; reason=None; child=None; accounting_error=None; incomplete=True
    with (out/'console.log').open('wb') as log:
        try:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while child.poll() is None:
                # Wall enforcement never depends on /proc accounting succeeding.
                wall=time.monotonic()-start
                if wall>=cfg['wall_seconds']:
                    reason='wall_budget'; kill_live(child); break
                try:
                    cpu,rss=sample(child.pid); own_cpu,own_rss=sample(os.getpid())
                except (OSError,ValueError,IndexError) as exc:
                    if child.poll() is not None:
                        append(out/'resources.jsonl',{'wall_seconds':time.monotonic()-start,
                            'event':'verified_child_exit_during_sample','sample_error':repr(exc)})
                        break
                    reason='accounting_failure'; accounting_error=repr(exc)
                    append(out/'resources.jsonl',{'wall_seconds':time.monotonic()-start,
                        'event':reason,'sample_error':accounting_error})
                    kill_live(child); break
                wall=time.monotonic()-start; peak=max(peak,rss+own_rss)
                append(out/'resources.jsonl',{'wall_seconds':wall,'child_cpu_seconds':cpu,'combined_rss_bytes':rss+own_rss})
                if wall>=cfg['wall_seconds']: reason='wall_budget'
                if cpu>=cfg['cpu_seconds']-5: reason='cpu_budget'
                if rss+own_rss>cfg['rss_bytes']: reason='sampled_rss_budget'
                if reason:
                    kill_live(child); break
                time.sleep(min(cfg['sampling_seconds'],max(0,cfg['wall_seconds']-(time.monotonic()-start))))
            code=child.wait()
            existing=json.loads((out/'status.json').read_text())
            incomplete=bool(reason or code or existing.get('status')!='complete')
            if incomplete:
                reason=reason or 'worker_failed_or_missing_completion'
                atomic(out/'status.json',{'status':'incomplete','reason':reason,
                    'returncode':code,'worker_status':existing,'accounting_error':accounting_error})
            atomic(out/'resource_summary.json',{'sampled_peak_combined_rss_bytes':peak,
                'wall_seconds':time.monotonic()-start,'returncode':code,'stop_reason':reason})
        except BaseException as exc:
            if child is not None:
                kill_live(child); child.wait()
            atomic(out/'status.json',{'status':'interrupted','error':repr(exc)})
            raise
        finally:
            atomic(out/'RESULTS-MANIFEST.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*')
                if p.is_file() and p.name!='RESULTS-MANIFEST.json'})
    if not incomplete: return 0
    return 124 if reason in ('wall_budget','cpu_budget','sampled_rss_budget') else (code if code>0 else 125)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--approved-freeze-sha256',required=True,help='Expected FREEZE.json SHA256')
    a=p.parse_args(); verify_freeze(ROOT)
    if a.approved_freeze_sha256!=digest(ROOT/'FREEZE.json'): raise SystemExit('Approval must name the exact frozen source')
    cfg=json.loads((ROOT/'protocol.json').read_text()); out=reserve(a.out)
    env=dict(os.environ,ADAPTIVE_SUPERVISED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             NUMEXPR_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    return monitor([sys.executable,str(ROOT/'runner.py'),'--out',str(out)],out,cfg,env)

if __name__=='__main__': raise SystemExit(main())
