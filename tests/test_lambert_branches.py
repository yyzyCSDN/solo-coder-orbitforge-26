import math
import pytest
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.errors import NoSolutionError, ValidationError
from orbitforge.maneuvers.lambert import (LambertReport, lambert_branches,
                                          lambert_transfer, lambert_universal)

MU = MU_EARTH_KM3_S2
R1 = Vec3(7000, 0, 0)
R2 = Vec3(0, 8000, 0)  # r1 x r2 along +z: short arc is prograde


def tof_from_states(r1, v1, r2, v2, revs, mu=MU):
    """Independent time of flight from the solved endpoint velocities,
    via Kepler's equation (elliptic) or its hyperbolic analogue."""
    def elements(r, v):
        a = -mu / (2.0 * (v.norm() ** 2 / 2.0 - mu / r.norm()))
        evec = v.cross(r.cross(v)) * (1.0 / mu) - r.unit()
        return a, evec.norm()
    a1, e1 = elements(r1, v1)
    a2, e2 = elements(r2, v2)
    assert abs(a1 - a2) < 1e-6 * abs(a1)  # both ends on the same conic
    assert e1 > 1e-6
    if a1 > 0.0:
        n = math.sqrt(mu / a1 ** 3)
        def mean_anomaly(r, v):
            E = math.atan2(r.dot(v) / (e1 * math.sqrt(mu * a1)),
                           (1.0 - r.norm() / a1) / e1)
            return E - e1 * math.sin(E)
        dm = (mean_anomaly(r2, v2) - mean_anomaly(r1, v1)) % (2.0 * math.pi)
        return (dm + 2.0 * math.pi * revs) / n
    n = math.sqrt(mu / (-a1) ** 3)
    def hyperbolic_anomaly(r, v):
        f = math.asinh(r.dot(v) / (e1 * math.sqrt(-mu * a1)))
        return e1 * math.sinh(f) - f
    return (hyperbolic_anomaly(r2, v2) - hyperbolic_anomaly(r1, v1)) / n


def check_solution(sol, dt, revs):
    assert sol.revolutions == revs
    assert abs(tof_from_states(R1, sol.v1, R2, sol.v2, revs) - dt) < 1e-3
    h = R1.cross(sol.v1)
    if sol.direction == 'prograde':
        assert h.z > 0
    else:
        assert h.z < 0
    if sol.arc == 'short':
        assert sol.transfer_angle < math.pi
    else:
        assert sol.transfer_angle > math.pi


def test_textbook_curtis_example():
    # Curtis, Orbital Mechanics for Engineering Students, Example 5.2.
    r1 = Vec3(5000, 10000, 2100)
    r2 = Vec3(-14600, 2500, 7000)
    v1, v2 = lambert_universal(r1, r2, 3600)
    assert (v1 - Vec3(-5.99249, 1.92536, 3.24564)).norm() < 1e-4
    assert (v2 - Vec3(-3.31246, -4.19662, -0.38529)).norm() < 1e-4


def test_short_and_long_arc_are_distinct_solutions():
    short = lambert_transfer(R1, R2, 1800, arc='short', direction='prograde')
    long = lambert_transfer(R1, R2, 1800, arc='long', direction='retrograde')
    assert short.arc == 'short' and short.direction == 'prograde'
    assert long.arc == 'long' and long.direction == 'retrograde'
    assert (short.v1 - long.v1).norm() > 1.0
    check_solution(short, 1800, 0)
    check_solution(long, 1800, 0)


def test_multi_rev_low_and_high_paths():
    dt = 12000.0
    low = lambert_transfer(R1, R2, dt, revolutions=1, path='low')
    high = lambert_transfer(R1, R2, dt, revolutions=1, path='high')
    assert low.path == 'low' and high.path == 'high'
    assert high.semimajor_axis > low.semimajor_axis
    check_solution(low, dt, 1)
    check_solution(high, dt, 1)


def test_multi_rev_two_revolutions():
    dt = 14000.0
    for path in ('low', 'high'):
        sol = lambert_transfer(R1, R2, dt, revolutions=2, path=path)
        check_solution(sol, dt, 2)


def test_below_branch_minimum_is_explicit_no_solution():
    rep = lambert_branches(R1, R2, 1800, max_revolutions=1,
                           arc='short', direction='prograde')
    branch = next(b for b in rep.branches if b.revolutions == 1)
    assert not branch.feasible and branch.solutions == ()
    assert branch.min_tof > 1800
    assert 'minimum' in branch.reason
    # Single-solution API raises instead of substituting the M=0 branch.
    with pytest.raises(NoSolutionError):
        lambert_transfer(R1, R2, 1800, revolutions=1, path='low')


def test_degenerate_solution_at_branch_minimum():
    rep = lambert_branches(R1, R2, 12000, max_revolutions=1,
                           arc='short', direction='prograde')
    tmin = next(b for b in rep.branches if b.revolutions == 1).min_tof
    sol = lambert_transfer(R1, R2, tmin, revolutions=1)
    assert sol.degenerate and sol.path == 'minimum'
    check_solution(sol, tmin, 1)


def test_inconsistent_arc_direction_not_substituted():
    # The short arc is prograde for this geometry; asking for a
    # retrograde short arc must fail loudly, not return the long arc.
    with pytest.raises(NoSolutionError):
        lambert_transfer(R1, R2, 1800, arc='short', direction='retrograde')
    rep = lambert_branches(R1, R2, 1800, arc='short', direction='retrograde')
    assert rep.solutions == ()
    assert all('inconsistent' in b.reason for b in rep.branches)


def test_ambiguous_request_requires_narrowing():
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, 12000, revolutions=1)  # low and high both match
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, 1800, arc='any', direction='any')


def test_singular_transfer_angles_raise():
    with pytest.raises(NoSolutionError, match='180'):
        lambert_universal(R1, Vec3(-14000, 0, 0), 1800)
    with pytest.raises(NoSolutionError, match='singular'):
        lambert_universal(R1, Vec3(14000, 0, 0), 1800)
    with pytest.raises(NoSolutionError):
        lambert_branches(R1, Vec3(-14000, 0, 0), 1800, max_revolutions=2)


def test_near_singular_solution_is_flagged():
    r2 = Vec3(-8000, 8000 * 1e-7, 0)  # ~1e-7 rad off the 180 deg singularity
    sol = lambert_transfer(R1, r2, 5000)
    assert sol.near_singular
    with pytest.raises(NoSolutionError):
        lambert_transfer(R1, Vec3(-8000, 8000 * 1e-13, 0), 5000)


def test_zero_rev_hyperbolic_leg():
    sol = lambert_transfer(R1, R2, 600)
    assert sol.semimajor_axis < 0
    check_solution(sol, 600, 0)


def test_report_accounts_for_every_requested_branch():
    rep = lambert_branches(R1, R2, 12000, max_revolutions=2,
                           arc='any', direction='any')
    assert isinstance(rep, LambertReport)
    # 2 arcs x 2 directions x 3 revolution counts, none missing.
    assert len(rep.branches) == 12
    for b in rep.branches:
        assert b.feasible or b.reason
    feasible = {(b.revolutions, b.arc, b.direction)
                for b in rep.branches if b.feasible}
    assert (0, 'short', 'prograde') in feasible
    assert (1, 'long', 'retrograde') in feasible
    assert (2, 'short', 'prograde') not in feasible  # dt below M=2 minimum


def test_legacy_wrapper_matches_new_api():
    v1, v2 = lambert_universal(R1, R2, 1800)
    sol = lambert_transfer(R1, R2, 1800)
    assert (v1 - sol.v1).norm() < 1e-12 and (v2 - sol.v2).norm() < 1e-12
    v1r, _ = lambert_universal(R1, R2, 1800, prograde=False)
    assert R1.cross(v1r).z < 0


def test_invalid_inputs_rejected():
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, -10)
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, 1800, arc='sideways')
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, 1800, path='middle')
