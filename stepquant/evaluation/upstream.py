"""Fetch only pinned evaluator source; never install a second CUDA/PyTorch stack."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

LOCK=Path(__file__).resolve().parents[2]/'configs/evaluation/upstream.lock.json'


def checkout(name, root):
    entry=json.loads(LOCK.read_text())[name]
    path=Path(root)/name
    if not (path/'.git').exists():
        subprocess.run(['git','clone','--depth','1','--no-checkout',entry['url'],str(path)],check=True)
    current=subprocess.run(['git','rev-parse','HEAD'],cwd=path,text=True,capture_output=True)
    dirty=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=path,text=True,capture_output=True,check=True)
    if dirty.stdout.strip():
        raise ValueError(f'official evaluator has local tracked changes: {path}')
    if current.stdout.strip()!=entry['revision']:
        subprocess.run(['git','fetch','origin',entry['revision'],'--depth','1'],cwd=path,check=True)
        subprocess.run(['git','checkout','--detach',entry['revision']],cwd=path,check=True)
    return path


def use(name, root):
    path=checkout(name,root)
    sys.path.insert(0,str(path/'src' if name=='matharena' else path))
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='artifacts/upstream')
    a=p.parse_args()
    for name in json.loads(LOCK.read_text()):
        print(checkout(name,a.root))


if __name__=='__main__':
    main()
