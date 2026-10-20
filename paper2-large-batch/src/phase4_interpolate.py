"""Run unmodified Phase 3 interpolation algorithm with Phase 4 architectures.
Uses the existing 61-point, BN-recalibration experiment implementation.
"""
import interpolate
from phase4_models import create

if __name__ == '__main__':
    interpolate.create = create
    interpolate.main()
