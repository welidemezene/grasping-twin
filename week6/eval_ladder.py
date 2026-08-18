import glob, os, subprocess, csv



CKPT_DIR = '/home/woldemedihn/grasping_twin/week4/checkpoints'
STAGE = 'stage23'
REPO = '/home/woldemedihn/grasping_twin'
IMAGE = 'grasping-twin-isaaclab:latest'

paths = glob.glob(os.path.join(CKPT_DIR, f'{STAGE}_*_steps.zip'))


def step_of(path):
    name = os.path.basename(path)
    return int(name.split('_')[1])


def ckpt_arg(path):
    name = os.path.basename(path)
    return f'checkpoints/{name[:-4]}'


def parse_probe(txt_path):
    coverage = None
    joints = None
    verdict = None
    for line in open(txt_path):
        if 'overall lifted' in line:
            coverage = float(line.split('(')[1].split('%')[0])
        elif 'on failures' in line:
            joints = float(line.split('on failures')[1].strip())
        elif 'VERDICT:' in line:
            verdict = 'KINEMATIC' if 'KINEMATIC' in line else 'POLICY'
    return coverage, joints, verdict


def run_probe(path):
    label = f'ladder_{step_of(path)}'
    cmd = [
        'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{REPO}:/workspace',
        '-w', '/workspace/week4',
        IMAGE,
        'probe_envelope.py',
        ckpt_arg(path),
        label,
        '--range', '0.15',
    ]
    print('running', ckpt_arg(path))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('FAILED:', ckpt_arg(path))
        print(r.stderr[-500:])
        return None
    return f'{REPO}/week4/envelope_{label}.txt'


paths.sort(key=step_of)
print(len(paths), 'checkpoints')

out_csv = f'{REPO}/week6/{STAGE}_ladder.csv'

with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'coverage', 'joints_on_fail', 'verdict'])

    for p in paths:
        txt = run_probe(p)
        if txt is None:
            continue
        coverage, joints, verdict = parse_probe(txt)
        w.writerow([step_of(p), coverage, joints, verdict])
        f.flush()
        print(step_of(p), coverage, verdict, flush=True)

print('wrote', out_csv)

