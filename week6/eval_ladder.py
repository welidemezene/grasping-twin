import glob, os, subprocess

CKPT_DIR = '/home/woldemedihn/grasping_twin/week4/checkpoints'
STAGE = 'stage23'
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

paths.sort(key=step_of)


print(len(paths), 'checkpoints')

print(parse_probe('/home/woldemedihn/grasping_twin/week4/envelope_ladder_test.txt'))

print(ckpt_arg(paths[14]))
