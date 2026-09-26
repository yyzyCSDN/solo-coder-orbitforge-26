"""Lambert's problem: time-fixed two-point orbital transfer.

Universal-variable (Stumpff function) solver with the full branch
structure of the problem made explicit:

* arc: 'short' (transfer angle < 180 deg) or 'long' (> 180 deg);
* direction: 'prograde' or 'retrograde', labelled by the sign of the
  angular momentum along +z.  Arc and direction are linked by the
  endpoint geometry (the short and long arcs have opposite orbit
  normals), so a requested combination can be inconsistent; that is
  reported explicitly instead of silently substituting another arc;
* revolutions: for M >= 1 a branch exists only above a minimum time of
  flight.  Below it there is no solution, exactly at it the two
  solutions coalesce into one degenerate solution, and above it there
  are two, labelled 'low' and 'high' path by semimajor axis.

Every root solve is bracketed inside the z-interval of its own branch,
so the iteration can never wander onto another branch.  Singular
transfer angles (0/180 deg, endpoints collinear with the focus, transfer
plane undefined) raise NoSolutionError.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.errors import NoSolutionError, ValidationError
from orbitforge.optimization.golden_section import minimize

_TWO_PI = 2.0 * math.pi
# |sin(dtheta)| at or below this: endpoints collinear, plane undefined.
_SINGULAR_SIN_TOL = 1e-12
# |sin(dtheta)| below this the plane is numerically ill-conditioned;
# solutions are still computed but flagged near_singular.
_NEAR_SINGULAR_SIN_TOL = 1e-6
# dt within this relative tolerance of a branch minimum: degenerate root.
_DEGEN_RTOL = 1e-9
_T_TOL = 1e-9       # seconds, root convergence in time of flight
_Z_TOL = 1e-13      # relative bracket width, root convergence in z
_MIN_Z_TOL = 1e-10  # absolute tolerance when locating the branch minimum

_ARCS = ('short', 'long')
_DIRECTIONS = ('prograde', 'retrograde')
_PATHS = ('low', 'high')


@dataclass(frozen=True)
class LambertSolution:
    """One feasible Lambert transfer, with its geometry labels."""
    v1: Vec3                 # departure velocity at r1 [km/s]
    v2: Vec3                 # arrival velocity at r2 [km/s]
    revolutions: int         # complete revolutions M on the transfer
    arc: str                 # 'short' (dtheta < pi) or 'long' (dtheta > pi)
    direction: str           # 'prograde' or 'retrograde' (sign of h_z)
    path: str                # 'single' (M=0), 'low', 'high' or 'minimum'
    transfer_angle: float    # geometric transfer angle dtheta [rad]
    semimajor_axis: float    # of the transfer conic [km], < 0 if hyperbolic
    degenerate: bool = False  # dt sits on the branch minimum (double root)
    near_singular: bool = False  # transfer angle close to 0/180 deg


@dataclass(frozen=True)
class LambertBranch:
    """Outcome for one requested (arc, direction, revolutions) branch."""
    revolutions: int
    arc: str
    direction: str
    solutions: tuple = ()     # tuple of LambertSolution, empty if infeasible
    reason: str | None = None  # why the branch has no solution
    min_tof: float | None = None  # branch minimum time of flight (M >= 1)

    @property
    def feasible(self):
        return bool(self.solutions)


@dataclass(frozen=True)
class LambertReport:
    """All requested branches; every branch is either feasible or
    carries an explicit reason, never a substituted solution."""
    dt: float
    mu: float
    branches: tuple

    @property
    def solutions(self):
        return tuple(s for b in self.branches for s in b.solutions)


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


def _tof(z, A, rsum, mu):
    """(time of flight, y) at universal z, or None outside the valid
    domain of this branch (y < 0, or c = 0 at a pole where tof -> inf)."""
    try:
        c = _stumpff_c(z)
        s = _stumpff_s(z)
    except OverflowError:
        return None
    if c <= 0.0:
        return None
    y = rsum + A * (z * s - 1.0) / math.sqrt(c)
    if y < 0.0:
        return None
    x = math.sqrt(y / c)
    return (x ** 3 * s + A * math.sqrt(y)) / math.sqrt(mu), y


def _tof_scalar(z, A, rsum, mu):
    r = _tof(z, A, rsum, mu)
    return math.inf if r is None else r[0]


def _root_z(zlo, zhi, dt, A, rsum, mu):
    """Root of tof(z) = dt inside [zlo, zhi].

    Safeguarded secant with bisection fallback; the iterate never
    leaves the bracket, so the result is guaranteed to belong to the
    branch the bracket was built for.  Pole endpoints (tof = inf) are
    handled as ordinary bracket endpoints.
    """
    def f(z):
        return _tof_scalar(z, A, rsum, mu) - dt
    flo = f(zlo)
    fhi = f(zhi)
    z = 0.5 * (zlo + zhi)
    for _ in range(200):
        if math.isfinite(flo) and math.isfinite(fhi) and flo != fhi:
            z = zhi - fhi * (zhi - zlo) / (fhi - flo)
            if not zlo < z < zhi:
                z = 0.5 * (zlo + zhi)
        else:
            z = 0.5 * (zlo + zhi)
        fz = f(z)
        if abs(fz) <= _T_TOL * max(1.0, dt):
            return z
        if zhi - zlo <= _Z_TOL * max(1.0, abs(z)):
            return z
        if (fz > 0.0) == (flo > 0.0):
            zlo, flo = z, fz
        else:
            zhi, fhi = z, fz
    return z


def _zero_rev_bracket(A, rsum, mu, dt):
    """Bracket the M = 0 root: (z_lower, (2 pi)^2), with tof < dt at the
    lower end.  The lower end is either z = 0, a negative (hyperbolic) z,
    or the y = 0 domain boundary where tof -> 0."""
    zhi = _TWO_PI ** 2
    if _tof_scalar(0.0, A, rsum, mu) < dt:
        return 0.0, zhi
    prev = 0.0
    for k in range(60):
        z = -2.0 ** k
        r = _tof(z, A, rsum, mu)
        if r is None:
            # y dropped below zero between z and prev; the domain
            # boundary (tof -> 0) lies in between.  Bisect for it.
            lo, hi = z, prev
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if _tof(mid, A, rsum, mu) is None:
                    lo = mid
                else:
                    hi = mid
            return hi, prev
        if r[0] < dt:
            return z, prev
        prev = z
    raise NoSolutionError('could not bracket the zero-revolution solution')


def _solve_branch(r1n, r2n, A, dt, mu, revolutions):
    """Roots for one (geometry, M) branch.

    Returns (roots, reason, min_tof) with roots a list of
    (z, y, path, degenerate).  reason is set when the branch has no
    solution; min_tof is the branch minimum time of flight for M >= 1.
    """
    rsum = r1n + r2n
    if revolutions == 0:
        zlo, zhi = _zero_rev_bracket(A, rsum, mu, dt)
        z = _root_z(zlo, zhi, dt, A, rsum, mu)
        y = _tof(z, A, rsum, mu)[1]
        return [(z, y, 'single', False)], None, None
    # M >= 1: tof(z) -> inf at both poles (2 pi M)^2 and (2 pi (M+1))^2
    # with a single minimum in between.
    zlo = (_TWO_PI * revolutions) ** 2
    zhi = (_TWO_PI * (revolutions + 1)) ** 2
    zmin, tmin = minimize(
        lambda z: _tof_scalar(z, A, rsum, mu), zlo, zhi, tolerance=_MIN_Z_TOL)
    if dt < tmin * (1.0 - _DEGEN_RTOL):
        return [], ('no solution: time of flight %.6g s is below the minimum '
                    '%.6g s for %d revolution(s) on this arc'
                    % (dt, tmin, revolutions)), tmin
    if abs(dt - tmin) <= tmin * _DEGEN_RTOL:
        # Double root at the branch minimum: the low and high paths
        # coalesce into one degenerate solution.
        y = _tof(zmin, A, rsum, mu)[1]
        return [(zmin, y, 'minimum', True)], None, tmin
    zl = _root_z(zlo, zmin, dt, A, rsum, mu)
    zr = _root_z(zmin, zhi, dt, A, rsum, mu)
    yl = _tof(zl, A, rsum, mu)[1]
    yr = _tof(zr, A, rsum, mu)[1]
    # Label the paths by energy: a = x^2 / z with x^2 = y / c.
    al = yl / (_stumpff_c(zl) * zl)
    ar = yr / (_stumpff_c(zr) * zr)
    low, high = ((zl, yl), (zr, yr)) if al <= ar else ((zr, yr), (zl, yl))
    return [(low[0], low[1], 'low', False),
            (high[0], high[1], 'high', False)], None, tmin


def _geometry_options(r1, r2):
    """Both arc geometries for the endpoint pair.

    Returns (r1n, r2n, options) with each option
    (arc, direction, transfer_angle, near_singular).  The short arc
    rotates r1 onto r2 about +(r1 x r2), the long arc about -(r1 x r2);
    the two arcs therefore always carry opposite direction labels.
    Raises NoSolutionError for collinear endpoints (0/180 deg).
    """
    r1n = r1.norm()
    r2n = r2.norm()
    if r1n <= 0.0 or r2n <= 0.0:
        raise ValidationError('endpoint position vectors must be non-zero')
    cross = r1.cross(r2)
    sin_g = cross.norm() / (r1n * r2n)
    cos_g = max(-1.0, min(1.0, r1.dot(r2) / (r1n * r2n)))
    if sin_g <= _SINGULAR_SIN_TOL:
        angle = '0' if cos_g > 0.0 else '180'
        raise NoSolutionError(
            'singular transfer angle (~%s deg): endpoints are collinear with '
            'the focus and the transfer plane is undefined' % angle)
    # atan2 keeps the angle accurate near pi, where acos loses digits.
    dtheta = math.atan2(sin_g, cos_g)
    near = sin_g < _NEAR_SINGULAR_SIN_TOL
    short_dir = 'prograde' if cross.z >= 0.0 else 'retrograde'
    long_dir = 'retrograde' if cross.z >= 0.0 else 'prograde'
    return r1n, r2n, (
        ('short', short_dir, dtheta, near),
        ('long', long_dir, _TWO_PI - dtheta, near),
    )


def _make_solution(r1, r2, r1n, r2n, mu, geom, revolutions, root):
    arc, direction, dtheta, near = geom
    z, y, path, degenerate = root
    A = math.sin(dtheta) * math.sqrt(r1n * r2n / (1.0 - math.cos(dtheta)))
    f = 1.0 - y / r1n
    g = A * math.sqrt(y / mu)
    gd = 1.0 - y / r2n
    v1 = (r2 - r1 * f) / g
    v2 = (r2 * gd - r1) / g
    energy = 0.5 * v1.dot(v1) - mu / r1n
    a = -mu / (2.0 * energy) if energy != 0.0 else math.inf
    return LambertSolution(v1=v1, v2=v2, revolutions=revolutions, arc=arc,
                           direction=direction, path=path,
                           transfer_angle=dtheta, semimajor_axis=a,
                           degenerate=degenerate, near_singular=near)


def _validate(dt, mu, max_revolutions, arc, direction):
    if dt <= 0.0:
        raise ValidationError('time of flight must be positive')
    if mu <= 0.0:
        raise ValidationError('gravitational parameter must be positive')
    if max_revolutions < 0:
        raise ValidationError('max_revolutions must be >= 0')
    if arc not in _ARCS + ('any',):
        raise ValidationError("arc must be 'short', 'long' or 'any', got %r" % (arc,))
    if direction not in _DIRECTIONS + ('any',):
        raise ValidationError(
            "direction must be 'prograde', 'retrograde' or 'any', got %r" % (direction,))


def lambert_branches(r1: Vec3, r2: Vec3, dt: float,
                     mu: float = MU_EARTH_KM3_S2,
                     max_revolutions: int = 0,
                     arc: str = 'any',
                     direction: str = 'any') -> LambertReport:
    """Enumerate the requested Lambert branches.

    Every requested (arc, direction, revolutions) combination appears
    in the report exactly once per revolution count: feasible branches
    carry their solutions, infeasible or geometrically inconsistent
    ones carry an explicit reason.  No branch is ever silently
    replaced by another one.
    """
    _validate(dt, mu, max_revolutions, arc, direction)
    r1n, r2n, options = _geometry_options(r1, r2)
    arcs = _ARCS if arc == 'any' else (arc,)
    directions = _DIRECTIONS if direction == 'any' else (direction,)
    branches = []
    for arc_label in arcs:
        for dir_label in directions:
            geom = next((o for o in options
                         if o[0] == arc_label and o[1] == dir_label), None)
            if geom is None:
                actual = next(o[1] for o in options if o[0] == arc_label)
                reason = ("inconsistent request: the %s arc is %s for this "
                          "endpoint geometry; no %s solution exists on it"
                          % (arc_label, actual, dir_label))
                for m in range(max_revolutions + 1):
                    branches.append(LambertBranch(m, arc_label, dir_label,
                                                  (), reason, None))
                continue
            dtheta = geom[2]
            A = math.sin(dtheta) * math.sqrt(r1n * r2n / (1.0 - math.cos(dtheta)))
            for m in range(max_revolutions + 1):
                roots, reason, tmin = _solve_branch(r1n, r2n, A, dt, mu, m)
                sols = tuple(_make_solution(r1, r2, r1n, r2n, mu, geom, m, root)
                             for root in roots)
                branches.append(LambertBranch(m, arc_label, dir_label,
                                              sols, reason, tmin))
    return LambertReport(dt, mu, tuple(branches))


def lambert_transfer(r1: Vec3, r2: Vec3, dt: float,
                     mu: float = MU_EARTH_KM3_S2,
                     revolutions: int = 0,
                     arc: str = 'any',
                     direction: str = 'prograde',
                     path: str = 'any') -> LambertSolution:
    """Solve for exactly one Lambert branch.

    Raises NoSolutionError when the requested branch has no solution
    (below the multi-revolution minimum time of flight, or an
    arc/direction combination that is inconsistent with the geometry);
    the answer never falls back to another branch.  Raises
    ValidationError when the request matches more than one solution
    and does not say which one is wanted.
    """
    if path not in _PATHS + ('any',):
        raise ValidationError("path must be 'low', 'high' or 'any', got %r" % (path,))
    report = lambert_branches(r1, r2, dt, mu, max_revolutions=revolutions,
                              arc=arc, direction=direction)
    branches = [b for b in report.branches if b.revolutions == revolutions]
    sols = [s for b in branches for s in b.solutions]
    if not sols:
        reasons = '; '.join(b.reason for b in branches if b.reason)
        raise NoSolutionError(
            'no Lambert solution for revolutions=%d, arc=%r, direction=%r: %s'
            % (revolutions, arc, direction, reasons or 'branch not feasible'))
    if path != 'any':
        # A degenerate 'minimum' solution is the coalesced low/high
        # path and legitimately answers either request.
        sols = [s for s in sols if s.path == path or s.path == 'minimum']
        if not sols:
            raise NoSolutionError(
                'no %s-path solution for revolutions=%d, arc=%r, direction=%r'
                % (path, revolutions, arc, direction))
    if len(sols) > 1:
        raise ValidationError(
            'request matches %d solutions; narrow it with arc, direction '
            'and/or path' % len(sols))
    return sols[0]


def lambert_universal(r1: Vec3, r2: Vec3, dt: float,
                      mu: float = MU_EARTH_KM3_S2, prograde: bool = True):
    """Zero-revolution Lambert solve kept for backward compatibility.

    Equivalent to lambert_transfer(..., revolutions=0, arc='any',
    direction='prograde' if prograde else 'retrograde'); returns the
    (v1, v2) velocity tuple.
    """
    sol = lambert_transfer(r1, r2, dt, mu, revolutions=0, arc='any',
                           direction='prograde' if prograde else 'retrograde')
    return sol.v1, sol.v2
