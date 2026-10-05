#!/usr/bin/env python3
"""Standalone cron scheduler. No dependency on BotMonitor or bot imports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime

BOTS = ('ccexchange','ETFEnhancer','ETFEnhancerLT','momentum_master',
        'options_covered','options_secured','options_inverted','options_direct')
BASE = Path(__file__).resolve().parent


def python_for(root):
    for name in ('venv','.venv'):
        candidate=root/name/'bin/python'
        if candidate.is_file() and os.access(candidate,os.X_OK):
            return candidate
    raise FileNotFoundError(f'No executable venv/bin/python or .venv/bin/python in {root}')


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        if p.stat().st_uid != os.getuid():
            return None
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':
            return None
        args=(p/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        return dict(pid=int(pid),ppid=int(fields[1]),started=fields[19],args=args)
    except (FileNotFoundError,ProcessLookupError):
        return None


def matches(record, root):
    args=record['args']
    if not args or 'python' not in Path(args[0]).name.lower():return False
    if '-c' in args or '-m' in args or '--backtest' in args:return False
    script=next((a for a in args[1:] if a and not a.startswith('-')),'')
    if Path(script).name not in ('launcher.py','main.py','monitor.py'):return False
    path=Path(script)
    if not path.is_absolute():
        # Sandboxed browser/helper processes may deny cwd access. Inspect cwd
        # only after argv establishes a candidate Python bot script.
        cwd=record.get('cwd')
        if cwd is None:cwd=(Path('/proc')/str(record['pid'])/'cwd').resolve(strict=True)
        path=cwd/path
    return path.resolve().parent == root.resolve()


def find_processes(root):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:record=process(p.name)
        except PermissionError:
            # A process view we cannot inspect must not authorize a duplicate launch.
            raise RuntimeError('Process inspection denied; refusing scheduling action')
        try:
            if record and matches(record,root):found.append(record)
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:raise RuntimeError('Candidate bot process cwd is inaccessible; refusing scheduling action')
    return found


def log(base,bot,message):
    stamp=datetime.now().astimezone().isoformat(timespec='seconds')
    line=f'[{stamp}] {bot}: {message}'
    print(line,flush=True)
    with (base/'cron_logs'/f'{bot}.log').open('a') as handle:handle.write(line+'\n')


def start_bot(base,bot):
    root=base/bot
    existing=find_processes(root)
    if existing:
        log(base,bot,'Already running: '+','.join(str(p['pid']) for p in existing))
        return True
    python=python_for(root)
    if not (root/'launcher.py').is_file():raise FileNotFoundError('launcher.py missing')
    environment=dict(os.environ,PYTHONUNBUFFERED='1')
    with (base/'cron_logs'/f'{bot}.log').open('ab',buffering=0) as output:
        child=subprocess.Popen([str(python),'-u',str(root/'launcher.py')],cwd=root,
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            env=environment,start_new_session=True)
    (base/'pids'/f'{bot}.pid').write_text(str(child.pid)+'\n')
    log(base,bot,f'Launcher started with {python}; PID {child.pid}')
    time.sleep(1)
    if child.poll() is not None:
        (base/'pids'/f'{bot}.pid').unlink(missing_ok=True)
        log(base,bot,f'Launcher exited with code {child.returncode}; inspect this log')
        return child.returncode==0
    return True


def same_process(record):
    current=process(record['pid'])
    return current if current and current['started']==record['started'] else None


def signal_checked(record,sig,root):
    current=same_process(record)
    if current and matches(current,root):
        try:os.kill(current['pid'],sig)
        except ProcessLookupError:pass


def stop_bot(base,bot,timeout=15):
    root=base/bot
    targets=find_processes(root)
    if not targets:
        log(base,bot,'No matching launcher or worker process')
        (base/'pids'/f'{bot}.pid').unlink(missing_ok=True)
        return True
    # Stop supervisors first, then their workers. Never trust a stale PID file.
    targets.sort(key=lambda p:not any(Path(a).name=='launcher.py' for a in p['args']))
    for record in targets:signal_checked(record,signal.SIGTERM,root)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline and any(same_process(p) for p in targets):time.sleep(.1)
    for record in targets:signal_checked(record,signal.SIGKILL,root)
    time.sleep(.1)
    # Catch a worker created by a supervisor in the narrow shutdown race.
    remaining=find_processes(root)
    for record in remaining:signal_checked(record,signal.SIGTERM,root)
    if remaining:
        time.sleep(.5)
        for record in remaining:signal_checked(record,signal.SIGKILL,root)
        time.sleep(.1)
    still_running=find_processes(root)
    if still_running:
        log(base,bot,'ERROR: matching processes remain; retaining PID record')
        return False
    (base/'pids'/f'{bot}.pid').unlink(missing_ok=True)
    log(base,bot,'Stopped verified launcher and worker processes')
    return True


def check(base):
    ok=True
    for bot in BOTS:
        try:
            root=base/bot
            python=python_for(root)
            if not (root/'launcher.py').is_file():raise FileNotFoundError('launcher.py missing')
            result=subprocess.run([str(python),'-I','-c','import sys; print(sys.version.split()[0])'],
                                  capture_output=True,text=True,timeout=10,check=True)
            running=find_processes(root)
            print(f'{bot}: READY | Python {result.stdout.strip()} | {python} | '+
                  ('RUNNING '+','.join(str(p['pid']) for p in running) if running else 'STOPPED'))
        except Exception as exc:
            ok=False;print(f'{bot}: ERROR | {exc}')
    return 0 if ok else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','stop','check'))
    args=parser.parse_args()
    if args.action=='check':return check(BASE)
    (BASE/'pids').mkdir(exist_ok=True)
    (BASE/'cron_logs').mkdir(exist_ok=True)
    with (BASE/'pids/.market_scheduler.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another scheduler action is active; refusing overlap',file=sys.stderr);return 1
        ok=True
        for bot in BOTS:
            try:ok=(start_bot(BASE,bot) if args.action=='start' else stop_bot(BASE,bot)) and ok
            except Exception as exc:
                log(BASE,bot,f'ERROR: {exc}');ok=False
        log(BASE,'market_runner',f'{args.action} completed; success={ok}')
        return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
