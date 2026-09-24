"""Run tests against an explicit build without executing shared site .pth files.

Use python -I -S run_build_tests.py --triton-package BUILD/triton
  --dependency-site SITE_PACKAGES --ptxas PATH --libdevice PATH -- TESTS...
External installations remain external; no global environment is modified.
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    if not (sys.flags.isolated and sys.flags.no_site):
        raise SystemExit('Required: python -I -S (prevent shared .pth startup injection)')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triton-package', required=True, type=Path)
    parser.add_argument('--dependency-site', required=True, type=Path)
    parser.add_argument('--ptxas', required=True, type=Path)
    parser.add_argument('--libdevice', required=True, type=Path)
    parser.add_argument('--cuda-include', type=Path,
                        help='CUDA headers; defaults to the declared ptxas installation include directory')
    parser.add_argument('tests', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    package = args.triton_package.resolve(strict=True)
    extension = (package / '_C/libtriton.so').resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    # Append dependency files without site.addsitedir: that would execute .pth.
    sys.path[:0] = [str(package.parent), str(root)]
    sys.path.append(str(args.dependency_site.resolve(strict=True)))
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.environ['L_CORE_TEST_EXTENSION'] = str(extension)
    os.environ['TRITON_PTXAS_PATH'] = str(args.ptxas.resolve(strict=True))
    os.environ['TRITON_LIBDEVICE_PATH'] = str(args.libdevice.resolve(strict=True))
    cuda_include=(args.cuda_include or args.ptxas.resolve().parent.parent/'include').resolve(strict=True)
    if not (cuda_include/'cuda.h').is_file():
        raise SystemExit('CUDA header missing; provide --cuda-include')
    os.environ['C_INCLUDE_PATH']=str(cuda_include)
    import triton
    from triton._C import libtriton
    if (Path(triton.__file__).resolve().parent != package or
            Path(libtriton.__file__).resolve() != extension):
        raise SystemExit('Compiler identity mismatch; no tests executed')
    print('Verified Triton:', triton.__file__, flush=True)
    print('Verified extension:', libtriton.__file__, flush=True)
    import pytest
    tests = args.tests[1:] if args.tests[:1] == ['--'] else args.tests
    if not tests:
        raise SystemExit('Explicit test targets required')
    return pytest.main(tests)


if __name__ == '__main__':
    raise SystemExit(main())
