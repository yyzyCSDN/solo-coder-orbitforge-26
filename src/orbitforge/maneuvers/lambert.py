from __future__ import annotations
import math
from dataclasses import dataclass
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.errors import NoSolutionError, ValidationError
from orbitforge.orbits.elements import state_to_elements
from orbitforge.optimization.golden_section import minimize as _golden_minimize

_TWO_PI = 2.0 * math.pi
# |sin(transfer angle)| below this the position vectors are (anti-)parallel
# and the transfer plane is geometrically undefined.
_SINGULAR_TOL = 1e-8
_TOL = 1e-10

def _stumpff_c(z):
    if z > 1e-08:
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-08:
        return (math.cosh(math.sqrt(-z)) - 1) / -z
    return 0.5 - z / 24 + z * z / 720

def _stumpff_s(z):
    if z > 1e-08:
        return (math.sqrt(z) - math.sin(math.sqrt(z))) / z ** 1.5
    if z < -1e-08:
        return (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / (-z) ** 1.5
    return 1 / 6 - z / 120 + z * z / 5040

@dataclass(frozen=True)
class LambertSolution:
    v1: Vec3
    v2: Vec3
    revolutions: int
    arc: str
    motion: str
    branch: str
    transfer_angle_rad: float
    tof_s: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float

    @property
    def speed1(self):
        return self.v1.norm()

    @property
    def speed2(self):
        return self.v2.norm()

    @property
    def swept_angle_rad(self):
        return self.transfer_angle_rad + _TWO_PI * self.revolutions

    @property
    def label(self):
        return (f'arc={self.arc} motion={self.motion} '
                f'revolutions={self.revolutions} branch={self.branch}')

@dataclass(frozen=True)
class LambertResult:
    solutions: tuple
    notes: tuple

    def __iter__(self):
        return iter(self.solutions)

    def __len__(self):
        return len(self.solutions)

def _tof_and_y(z, r1n, r2n, A, mu):
    c = _stumpff_c(z)
    if c <= 0.0:
        return None
    s = _stumpff_s(z)
    y = r1n + r2n + A * (z * s - 1.0) / math.sqrt(c)
    if y <= 0.0:
        return None
    x = math.sqrt(y / c)
    return (x ** 3 * s + A * math.sqrt(y)) / math.sqrt(mu), y

def _eval_tof(z, r1n, r2n, A, mu):
    res = _tof_and_y(z, r1n, r2n, A, mu)
    if res is None:
        raise NoSolutionError('infeasible universal variable inside bracket')
    return res[0]

def _geometry(r1, r2, arc, singular_tol):
    r1n, r2n = r1.norm(), r2.norm()
    if r1n <= 0.0 or r2n <= 0.0:
        raise ValidationError('Lambert geometry requires nonzero position vectors')
    cosd = max(-1.0, min(1.0, r1.dot(r2) / (r1n * r2n)))
    base = math.acos(cosd)
    dtheta = _TWO_PI - base if arc == 'long' else base
    if abs(math.sin(dtheta)) < singular_tol:
        raise NoSolutionError(
            f'transfer angle {math.degrees(base):.8g} deg is degenerate: '
            'position vectors are (anti-)parallel, transfer plane undefined')
    sign = -1.0 if arc == 'long' else 1.0
    A = sign * math.sqrt(r1n * r2n * (1.0 + cosd))
    return r1n, r2n, dtheta, A

def _solve_z(r1n, r2n, A, mu, lo, hi, dt):
    # Illinois false-position on a bracket with opposite-sign residuals;
    # z never leaves [lo, hi], so the iteration cannot cross into another
    # revolution band.
    f_lo = _eval_tof(lo, r1n, r2n, A, mu) - dt
    f_hi = _eval_tof(hi, r1n, r2n, A, mu) - dt
    if f_lo * f_hi > 0.0:
        raise NoSolutionError('Lambert root is not bracketed')
    side = 0
    for _ in range(100):
        z = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        f_z = _eval_tof(z, r1n, r2n, A, mu) - dt
        if abs(f_z) <= _TOL * max(1.0, dt):
            return z
        if f_z * f_hi < 0.0:
            lo, f_lo = z, f_z
            if side == -1:
                f_hi *= 0.5
            side = -1
        else:
            hi, f_hi = z, f_z
            if side == 1:
                f_lo *= 0.5
            side = 1
    raise NoSolutionError('Lambert iteration did not converge')

def _walk_to_edge(r1n, r2n, A, mu, edge, start, dt):
    # Time of flight grows without bound toward a band edge; walk halfway
    # steps from start until it exceeds dt.
    z = start
    for _ in range(200):
        z = edge + 0.5 * (z - edge)
        if z == edge:
            break
        res = _tof_and_y(z, r1n, r2n, A, mu)
        if res is not None and res[0] > dt:
            return z
    raise NoSolutionError(f'could not bracket time of flight {dt} s near band edge')

def _bracket_zero_rev(r1n, r2n, A, mu, dt):
    t0 = _eval_tof(0.0, r1n, r2n, A, mu)
    if dt >= t0:
        return 0.0, _walk_to_edge(r1n, r2n, A, mu, _TWO_PI ** 2, 0.0, dt)
    # Hyperbolic side: time of flight falls to zero at the y = 0 boundary.
    z_above = 0.0
    z = -1.0
    for _ in range(200):
        res = _tof_and_y(z, r1n, r2n, A, mu)
        if res is None:
            z = 0.5 * (z + z_above)
            continue
        if res[0] < dt:
            return z, z_above
        z_above = z
        z *= 2.0
    raise NoSolutionError('could not bracket hyperbolic transfer')

def _band_minimum(r1n, r2n, A, mu, revolutions):
    lo = (_TWO_PI * revolutions) ** 2
    hi = (_TWO_PI * (revolutions + 1)) ** 2
    return _golden_minimize(lambda z: _eval_tof(z, r1n, r2n, A, mu), lo, hi, tolerance=1e-10)

def _build(r1, r2, r1n, r2n, A, dtheta, mu, revolutions, branch, z, dt):
    res = _tof_and_y(z, r1n, r2n, A, mu)
    if res is None:
        raise NoSolutionError('Lambert solution landed on infeasible geometry')
    y = res[1]
    f = 1.0 - y / r1n
    g = A * math.sqrt(y / mu)
    gd = 1.0 - y / r2n
    v1 = (r2 - r1 * f) / g
    v2 = (r2 * gd - r1) / g
    el = state_to_elements(r1, v1, mu)
    cosi = max(-1.0, min(1.0, math.cos(el.i_rad)))
    motion = 'prograde' if cosi > 1e-12 else 'retrograde' if cosi < -1e-12 else 'polar'
    return LambertSolution(v1=v1, v2=v2, revolutions=revolutions,
                           arc='short' if dtheta < math.pi else 'long',
                           motion=motion, branch=branch,
                           transfer_angle_rad=dtheta, tof_s=dt,
                           semi_major_axis_km=el.a_km, eccentricity=el.e,
                           inclination_rad=el.i_rad)

def _solve_geometry(r1, r2, r1n, r2n, dtheta, A, dt, mu, revolutions, arc):
    if revolutions == 0:
        lo, hi = _bracket_zero_rev(r1n, r2n, A, mu, dt)
        z = _solve_z(r1n, r2n, A, mu, lo, hi, dt)
        return [_build(r1, r2, r1n, r2n, A, dtheta, mu, 0, 'single', z, dt)], []
    z_min, t_min = _band_minimum(r1n, r2n, A, mu, revolutions)
    tol = 1e-9 * max(1.0, t_min)
    if dt < t_min - tol:
        return [], [f"arc='{arc}' revolutions={revolutions}: no solution, time of flight "
                    f'{dt:.6g} s is below the minimum {t_min:.6g} s for this branch']
    if dt <= t_min + tol:
        note = (f"arc='{arc}' revolutions={revolutions}: degenerate, time of flight equals "
                f"the minimum {t_min:.6g} s; 'low' and 'high' branches merge")
        return [_build(r1, r2, r1n, r2n, A, dtheta, mu, revolutions, 'minimum-time', z_min, dt)], [note]
    # 'high' = larger semi-major axis (z below the minimum-time point),
    # 'low' = smaller semi-major axis (z above it).
    z_edge_lo = _walk_to_edge(r1n, r2n, A, mu, (_TWO_PI * revolutions) ** 2, z_min, dt)
    z_edge_hi = _walk_to_edge(r1n, r2n, A, mu, (_TWO_PI * (revolutions + 1)) ** 2, z_min, dt)
    z_high = _solve_z(r1n, r2n, A, mu, z_edge_lo, z_min, dt)
    z_low = _solve_z(r1n, r2n, A, mu, z_min, z_edge_hi, dt)
    return [_build(r1, r2, r1n, r2n, A, dtheta, mu, revolutions, 'high', z_high, dt),
            _build(r1, r2, r1n, r2n, A, dtheta, mu, revolutions, 'low', z_low, dt)], []

def _validate(dt, mu, revolutions):
    if dt <= 0.0:
        raise ValidationError('time of flight must be positive')
    if mu <= 0.0:
        raise ValidationError('gravitational parameter must be positive')
    if not isinstance(revolutions, int) or revolutions < 0:
        raise ValidationError('revolutions must be a nonnegative integer')

def _check_motion(sol, motion, arc):
    if sol.motion != motion:
        other = 'long' if arc == 'short' else 'short'
        raise NoSolutionError(
            f"requested motion='{motion}' but the {arc} arc between these vectors is "
            f"{sol.motion}; use arc='{other}'")
    return sol

def lambert_transfer(r1: Vec3, r2: Vec3, dt: float, mu: float=MU_EARTH_KM3_S2,
                     revolutions: int=0, arc: str='short', branch: str=None,
                     motion: str=None, singular_tol: float=_SINGULAR_TOL) -> LambertSolution:
    _validate(dt, mu, revolutions)
    if arc not in ('short', 'long'):
        raise ValidationError("arc must be 'short' or 'long'")
    if motion not in (None, 'prograde', 'retrograde'):
        raise ValidationError("motion must be 'prograde' or 'retrograde'")
    if branch is None:
        if revolutions == 0:
            branch = 'single'
        else:
            raise ValidationError("multi-revolution transfers have two branches; "
                                  "pass branch='low' or branch='high'")
    r1n, r2n, dtheta, A = _geometry(r1, r2, arc, singular_tol)
    sols, notes = _solve_geometry(r1, r2, r1n, r2n, dtheta, A, dt, mu, revolutions, arc)
    for sol in sols:
        if sol.branch == branch:
            return _check_motion(sol, motion, arc) if motion else sol
    detail = '; '.join(notes) or f"branch '{branch}' does not exist for this geometry"
    raise NoSolutionError(f"no {arc}-arc {revolutions}-revolution solution on branch "
                          f"'{branch}': {detail}")

def lambert_solutions(r1: Vec3, r2: Vec3, dt: float, mu: float=MU_EARTH_KM3_S2,
                      max_revolutions: int=0, arc: str='both', motion: str='any',
                      singular_tol: float=_SINGULAR_TOL) -> LambertResult:
    _validate(dt, mu, max_revolutions)
    if arc == 'both':
        arcs = ('short', 'long')
    elif arc in ('short', 'long'):
        arcs = (arc,)
    else:
        raise ValidationError("arc must be 'short', 'long' or 'both'")
    if motion not in ('prograde', 'retrograde', 'any'):
        raise ValidationError("motion must be 'prograde', 'retrograde' or 'any'")
    solutions, notes = [], []
    for a in arcs:
        try:
            r1n, r2n, dtheta, A = _geometry(r1, r2, a, singular_tol)
        except NoSolutionError as exc:
            notes.append(f"arc='{a}': {exc}")
            continue
        for revs in range(max_revolutions + 1):
            sols, ns = _solve_geometry(r1, r2, r1n, r2n, dtheta, A, dt, mu, revs, a)
            notes.extend(ns)
            if revs >= 1 and not sols:
                notes.append(f"arc='{a}': revolution counts above {revs} are also infeasible "
                             '(minimum time of flight grows with revolutions)')
                break
            solutions.extend(sols)
    if motion != 'any':
        kept = [s for s in solutions if s.motion == motion]
        if not kept and solutions:
            notes.append(f'no {motion} branch among the feasible solutions')
        solutions = kept
    if not solutions:
        raise NoSolutionError('no feasible Lambert solution: ' + '; '.join(notes))
    return LambertResult(tuple(solutions), tuple(notes))

def lambert_min_tof(r1: Vec3, r2: Vec3, mu: float=MU_EARTH_KM3_S2,
                    revolutions: int=1, arc: str='short',
                    singular_tol: float=_SINGULAR_TOL) -> float:
    if not isinstance(revolutions, int) or revolutions < 1:
        raise ValidationError('minimum-time branches exist for revolutions >= 1')
    if arc not in ('short', 'long'):
        raise ValidationError("arc must be 'short' or 'long'")
    r1n, r2n, dtheta, A = _geometry(r1, r2, arc, singular_tol)
    return _band_minimum(r1n, r2n, A, mu, revolutions)[1]

def lambert_universal(r1: Vec3, r2: Vec3, dt: float, mu: float=MU_EARTH_KM3_S2, prograde: bool=True):
    cross_z = r1.cross(r2).z
    if prograde:
        arc = 'short' if cross_z >= 0 else 'long'
    else:
        arc = 'short' if cross_z < 0 else 'long'
    sol = lambert_transfer(r1, r2, dt, mu, revolutions=0, arc=arc)
    return (sol.v1, sol.v2)
