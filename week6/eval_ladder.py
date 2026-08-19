import glob, os, subprocess, csv, sys



CKPT_DIR = '/home/woldemedihn/grasping_twin/week4/checkpoints'
STAGE = 'stage23'
REPO = '/home/woldemedihn/grasping_twin'
IMAGE = 'grasping-twin-isaaclab:latest'
N_BELOW = 3

names = sys.argv[1:]
if names:
    paths = [os.path.join(CKPT_DIR, n) for n in names]
else:
    paths = glob.glob(os.path.join(CKPT_DIR, f'{STAGE}_*.zip'))



def step_of(path):
    name = os.path.basename(path)
    if 'final' in name:
        return 10000000
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
rows = []
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'coverage', 'joints_on_fail', 'verdict'])

    


    for p in paths:
        txt = run_probe(p)
        if txt is None:
            continue
        coverage, joints, verdict = parse_probe(txt)
        w.writerow([step_of(p), coverage, joints, verdict])
        rows.append([step_of(p), coverage, joints, verdict])


        f.flush()
        print(step_of(p), coverage, verdict, flush=True)

print('wrote', out_csv)
peak = max(rows, key=lambda row: row[1])
threshold = peak[1] * 0.9
overrun = rows[-1][0] - peak[0]

count = 0
streak_start = None
i = rows.index(peak)

for row in rows[i+1:]:
    if row[1] < threshold:              
        count = count + 1
        if count == 1:
            streak_start = row[0]    
    else:
        count = 0
        streak_start = None

    if count == N_BELOW:              
        print(f'peak rule fired at step {streak_start}')
        break  
if count < N_BELOW:
    print('peak rule never fired')



print(f'peak {peak[1]}% at step {peak[0]}')
print(f'threshold {threshold:.1f}%')
print(f'run continued {overrun} steps past the peak')

