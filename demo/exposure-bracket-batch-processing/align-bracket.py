# this is a script for using img-aligner to align exposure-bracketed images
# https://github.com/bean-mhm/img-aligner

from io import IOBase
import os
from pathlib import Path
import subprocess
import time

# path to the img-aligner executable
img_aligner_path = R'../../img-aligner'

# path to the directory containing the images, without an ending (back)slash.
# exposure-bracketed image files must start with 'EXP ' followed by a number
# representing the exposure time in seconds. they can also start with 'EXPR ' in
# which case the following number will represent the inverse (reciprocal) of the
# exposure time.
#
# example file names:
# - 'EXPR 5 forest.exr' (1/5 sec. exposure)
# - 'EXP 2 2025-04-24.png' (2 sec. exposure)
img_dir = R'.'

# suffix to add to the warped image file.
# example: 'EXPR 25 car.exr' -> 'EXPR 25 car (aligned).exr'
output_warped_img_path_suffix = ' (aligned)'

# additional arguments. use 'img-aligner --cli --help' to see all arguments.
extra_args = [
    '--cost-res', '240',
    '--interm-res', '2000000',
    '--warp-strength', '0.00015',
    '--warp-strength-decay', '0.001',
    '--min-warp-strength', '0.0001',
    '--min-change-in-cost', '0.000005'
]


class ImageFile:
    path: str
    exposure: float

    def __init__(
        self,
        path: str,
        exposure: float
    ):
        self.path = path
        self.exposure = exposure

    def __lt__(self, other):
        return self.exposure < other.exposure

    def __eq__(self, other):
        return \
            self.path == other.path \
            and self.exposure == other.exposure


# print IO stream output (with custom formatting)
def print_io(io: IOBase):
    s = ''
    while True:
        if io.closed:
            break

        char = io.read(1).decode()
        if len(char) < 1 or char == '\0':
            break

        s += char

    s = s.strip()
    if (s == ''):
        return

    # left padding
    s = '  > ' + s
    s = s.replace('\n', '\n  > ')

    # vertical padding
    print(f'\n{s}\n\n', end='')


# print IO stream output in realtime (with custom formatting)
def print_io_realtime(io: IOBase):
    print('\n  > ', end='')

    while True:
        if io.closed:
            break

        char = io.read(1).decode()
        if len(char) < 1 or char == '\0':
            break

        if char == '\n':
            print('\n  > ', end='')
        else:
            print(char, end='')

    print('\n\n', end='')


def run_and_print(args):
    # create subprocess
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    proc.stdin.write('\n'.encode())
    proc.stdin.flush()

    print_io_realtime(proc.stdout)
    print_io(proc.stderr)

    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=5)


all_start_time = time.time()

# list of image files we find
image_files: list[ImageFile] = []

# iterate the directory and find exposure-bracketed image files
for file in os.listdir(os.fsencode(img_dir)):
    path = os.fsdecode(file)

    if not (path.startswith('EXP ') or path.startswith('EXPR ')):
        continue

    # avoid recollecting previously warped images
    if Path(path).stem.endswith(output_warped_img_path_suffix):
        continue

    start_idx: int = 4
    end_idx: int = start_idx

    reciprocal = False
    if path.startswith('EXPR '):
        reciprocal = True
        start_idx += 1
        end_idx += 1

    while end_idx < len(path) and path[end_idx] in '0123456789':
        end_idx += 1

    if end_idx > len(path) or (end_idx - start_idx) < 1:
        continue

    exposure = float(path[start_idx:end_idx])
    if reciprocal:
        exposure = 1. / exposure

    image_files.append(ImageFile(
        f'{img_dir}/{path}',
        exposure
    ))

# sort the list ordered from the brightest to the darkest image
image_files.sort(reverse=True)

print(f'found {len(image_files)} exposure-bracketed images')
for image_file in image_files:
    print(f'- {Path(image_file.path).name} ({image_file.exposure:.8f} sec.)')
print('')

if len(image_files) < 2:
    print('need at least 2 images')
    exit()

# process the images
for i in range(1, len(image_files)):
    base = image_files[i]
    target = image_files[i - 1]

    base_img_path = Path(base.path)
    target_img_path = Path(target.path)
    output_warped_img_path = base_img_path.with_stem(
        base_img_path.stem + output_warped_img_path_suffix
    )

    # if the target image has a warped version, use that instead
    if i > 1:
        target_img_path = target_img_path.with_stem(
            target_img_path.stem + output_warped_img_path_suffix
        )

    print(f'aligning "{base_img_path.name}" to "{target_img_path.name}"')

    base_img_multiplier = target.exposure / base.exposure
    args = [
        img_aligner_path,
        '--cli',
        '-b', str(base_img_path),
        '-t', str(target_img_path),
        '-o', str(output_warped_img_path),
        '--base-mul', str(base_img_multiplier)
    ]
    args.extend(extra_args)

    # run
    start = time.time()
    run_and_print(args)
    end = time.time()

    print(f'"{base_img_path.name}" done in {(end - start):.2f} s\n')

all_end_time = time.time()
print(f'\nall images done in {(all_end_time - all_start_time):.2f} s\n')
