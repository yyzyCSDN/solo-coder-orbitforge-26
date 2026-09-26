import math
import pytest
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2 as MU
from orbitforge.core.errors import NoSolutionError, ValidationError
from orbitforge.maneuvers.lambert import (
    lambert_universal, lambert_transfer, lambert_solutions, lambert_min_tof)
from orbitforge.orbits.universal import propagate_universal

R1 = Vec3(7000, 0, 0)
R2 = Vec3(0, 8000, 0)

# Round-trip tolerance: the library universal propagator is only good to
# ~1e-6 km on multi-revolution ellipses; solver TOF error is ~1e-9 s.
def roundtrip_error(sol, r1=R1, r2=R2):
    r, v = propagate_universal(r1, sol.v1, sol.tof_s)
    return (r - r2).norm()

def energy_mismatch(sol, r1=R1, r2=R2):
    e1 = sol.v1.norm2() / 2 - MU / r1.norm()
    e2 = sol.v2.norm2() / 2 - MU / r2.norm()
    return abs(e1 - e2)

def test_legacy_single_rev_prograde():
    v1, v2 = lambert_universal(R1, R2, 1800)
    assert v1.y > 0 and v2.x < 0
    r, v = propagate_universal(R1, v1, 1800)
    assert (r - R2).norm() < 1e-5

def test_legacy_retrograde_goes_clockwise():
    v1, v2 = lambert_universal(R1, R2, 3600, prograde=False)
    assert R1.cross(v1).z < 0
    r, v = propagate_universal(R1, v1, 3600)
    assert (r - R2).norm() < 1e-5

def test_short_and_long_arc_labels():
    short = lambert_transfer(R1, R2, 3600, arc='short')
    long = lambert_transfer(R1, R2, 3600, arc='long')
    assert short.arc == 'short' and short.transfer_angle_rad < math.pi
    assert long.arc == 'long' and long.transfer_angle_rad > math.pi
    assert short.motion == 'prograde' and long.motion == 'retrograde'
    assert short.revolutions == 0 and short.branch == 'single'
    assert roundtrip_error(short) < 1e-5 and roundtrip_error(long) < 1e-5

def test_multi_rev_two_branches():
    res = lambert_solutions(R1, R2, 12000, max_revolutions=1, arc='short')
    assert len(res) == 3
    by_branch = {s.branch: s for s in res}
    assert set(by_branch) == {'single', 'low', 'high'}
    high, low = by_branch['high'], by_branch['low']
    assert high.revolutions == low.revolutions == 1
    assert high.semi_major_axis_km > low.semi_major_axis_km
    assert high.swept_angle_rad > 2 * math.pi
    for sol in res:
        assert roundtrip_error(sol) < 1e-5
        assert energy_mismatch(sol) < 1e-8

def test_multi_rev_two_revolutions():
    dt = 14000
    res = lambert_solutions(R1, R2, dt, max_revolutions=2, arc='short')
    revs = sorted(s.revolutions for s in res)
    assert revs == [0, 1, 1, 2, 2]
    for sol in res:
        assert roundtrip_error(sol) < 1e-5

def test_infeasible_multi_rev_is_explicit_not_substituted():
    dt = 1000.0
    with pytest.raises(NoSolutionError, match='minimum'):
        lambert_transfer(R1, R2, dt, revolutions=1, branch='low')
    res = lambert_solutions(R1, R2, dt, max_revolutions=3, arc='short')
    assert [s.revolutions for s in res] == [0]
    assert any('minimum' in n for n in res.notes)
    assert any('revolutions=1' in n for n in res.notes)

def test_minimum_time_degenerate():
    tmin = lambert_min_tof(R1, R2, revolutions=1, arc='short')
    assert 7000 < tmin < 8000
    res = lambert_solutions(R1, R2, tmin, max_revolutions=1, arc='short')
    degen = [s for s in res if s.branch == 'minimum-time']
    assert len(degen) == 1
    assert any('degenerate' in n for n in res.notes)
    assert roundtrip_error(degen[0]) < 1e-5
    with pytest.raises(NoSolutionError, match='degenerate'):
        lambert_transfer(R1, R2, tmin, revolutions=1, branch='low')

def test_singular_transfer_angles_raise():
    with pytest.raises(NoSolutionError, match='degenerate'):
        lambert_transfer(Vec3(7000, 0, 0), Vec3(-7000, 0, 0), 3600)
    with pytest.raises(NoSolutionError, match='degenerate'):
        lambert_transfer(Vec3(7000, 0, 0), Vec3(14000, 0, 0), 3600)
    near = math.pi - 1e-9
    r2 = Vec3(7000 * math.cos(near), 7000 * math.sin(near), 0)
    with pytest.raises(NoSolutionError, match='degenerate'):
        lambert_transfer(Vec3(7000, 0, 0), r2, 3600)
    with pytest.raises(NoSolutionError):
        lambert_universal(Vec3(7000, 0, 0), Vec3(-7000, 0, 0), 3600)
    with pytest.raises(NoSolutionError, match='degenerate'):
        lambert_solutions(Vec3(7000, 0, 0), Vec3(-7000, 0, 0), 3600, arc='both')

def test_motion_filter_and_mismatch():
    res = lambert_solutions(R1, R2, 3600, arc='both', motion='retrograde')
    assert len(res) == 1 and res.solutions[0].motion == 'retrograde'
    res = lambert_solutions(R1, R2, 3600, arc='both', motion='prograde')
    assert len(res) == 1 and res.solutions[0].motion == 'prograde'
    with pytest.raises(NoSolutionError, match='retrograde'):
        lambert_transfer(R1, R2, 3600, arc='short', motion='retrograde')

def test_motion_labels_3d_geometry():
    r1 = Vec3(7000, 1000, 500)
    r2 = Vec3(-3000, 6000, 2000)
    res = lambert_solutions(r1, r2, 4000, arc='both')
    assert len(res) == 2
    motions = {s.arc: s.motion for s in res}
    assert motions['short'] != motions['long']
    for sol in res:
        assert roundtrip_error(sol, r1, r2) < 1e-5
        h = r1.cross(sol.v1)
        cosi = h.z / h.norm()
        if sol.motion == 'prograde':
            assert cosi > 0
        elif sol.motion == 'retrograde':
            assert cosi < 0

def test_hyperbolic_short_time():
    sol = lambert_transfer(R1, R2, 900.0, arc='short')
    assert sol.eccentricity > 1
    assert sol.semi_major_axis_km < 0
    assert roundtrip_error(sol) < 1e-5

def test_labels_and_speeds():
    sol = lambert_transfer(R1, R2, 12000, revolutions=1, branch='high')
    assert sol.label == 'arc=short motion=prograde revolutions=1 branch=high'
    assert sol.speed1 == sol.v1.norm() and sol.speed2 == sol.v2.norm()
    assert sol.swept_angle_rad == sol.transfer_angle_rad + 2 * math.pi
    assert sol.inclination_rad < math.pi / 2

def test_invalid_requests():
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, -10)
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, 12000, revolutions=1)
    with pytest.raises(ValidationError):
        lambert_transfer(R1, R2, 3600, arc='sideways')
    with pytest.raises(ValidationError):
        lambert_solutions(R1, R2, 3600, max_revolutions=-1)
    with pytest.raises(ValidationError):
        lambert_min_tof(R1, R2, revolutions=0)
