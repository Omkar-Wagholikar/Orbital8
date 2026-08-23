# SIMD Gravity Sim

A tiny N-body gravity simulator, built as a hands-on way to learn AVX2 SIMD
intrinsics. It's real (if toy) physics — a sun, planets, and elliptical
orbits — rendered live with [raylib](https://www.raylib.com/), but the point
of the project is the vector math underneath it, not the simulation itself.

## Why gravity, specifically

N-body gravity is a good SIMD teaching example because the expensive part —
"sum up the pull of every other body on this one" — is a horizontal
reduction over a fixed-size array, and this project caps bodies at **8**,
exactly the width of one AVX2 `__m256` register (8 × 32-bit floats). That
means the inner loop of the simulation isn't "vectorize a loop over N
bodies" — it's "the 8 bodies just *are* one register." No loop, no
remainder handling, no masking. One vector holds all the x-positions, one
holds all the masses, and one FMA-shaped chain of instructions computes
every pairwise force at once.

## The physics

Each body pulls on every other body with Newtonian gravity. For a target
body `i` being pulled on by every other body `j`:

```
a_i = Σⱼ  m_j · (r_j − r_i) / |r_j − r_i|³
```

(`G` is set to 1 throughout — this sim cares about relative dynamics, not
real units.)

Two problems show up immediately if you implement that literally:

1. **Division by zero.** A body's distance to itself is 0, so `|r_j-r_i|³`
   is 0 for `j = i`.
2. **Near-collisions blow up.** Two bodies passing close to each other spike
   the acceleration toward infinity, which an Euler integrator can't
   survive — one body slingshots out of the simulation.

The standard fix is **Plummer softening**: add a small constant `η` (`eta`
in the code) inside the distance term before cubing it:

```
a_i = Σⱼ  m_j · (r_j − r_i) / (|r_j − r_i|² + η)^1.5
```

This caps the maximum force at close range instead of letting it diverge.
The tradeoff (discovered the hard way while adding a moon to this sim — see
[Lessons learned](#lessons-learned)) is that at very small separations the
`η` term dominates over the real squared-distance term, and the force
degenerates from inverse-square gravity into something closer to a spring
(`F ∝ r` instead of `F ∝ 1/r²`). That's fine at planet-to-sun scale; it
quietly breaks anything you try to bind tightly at sub-unit scale, like a
moon orbiting a planet.

One more thing worth calling out because it trips people up: **a body's own
mass never appears in its own acceleration.** `a_i` only depends on the
*other* bodies' masses (`m_j`) — this falls straight out of `F = Gm_im_j/r²`
divided by `a = F/m_i`, where the target's mass cancels. A feather and a
planet fall at the same rate in this sim for the same reason they do in
reality (Galileo's equivalence principle). Mass only matters for how hard a
body pulls on *others* — it's set to 0 for a body's rendering to skip it
entirely (see [`render_points`](#body-slots--rendering)), which is also how
this codebase represents an "empty" body slot.

## The SIMD implementation

### `fast_sum`: reducing 8 lanes to 1 float

```c
float fast_sum(__m256 ax0) {
    __m128 lo = _mm256_castps256_ps128(ax0);      // lanes 0-3
    __m128 hi = _mm256_extractf128_ps(ax0, 1);     // lanes 4-7
    __m128 s  = _mm_add_ps(lo, hi);                // pairwise fold 8→4
    s = _mm_hadd_ps(s, s);                         // horizontal add 4→2
    s = _mm_hadd_ps(s, s);                         // horizontal add 2→1
    return _mm_cvtss_f32(s);                       // extract lane 0
}
```

AVX2 has no single "sum all 8 lanes" instruction, so this is the standard
idiom: split the 256-bit register into two 128-bit halves, add them
together, then use `_mm_hadd_ps` twice to fold 4 lanes down to 1. Every lane
of the result ends up holding the same total; `_mm_cvtss_f32` just reads it
out as a scalar.

### `get_instantenout_acceletation`: one body's pull from all 8 sources

Given a target position `(s_x, s_y)` and the positions/masses of all 8
bodies as `__m256` vectors, this computes the target's acceleration in one
pass:

```c
__m256 rx = _mm256_sub_ps(x, x0);          // (source_x − target_x), all 8 at once
__m256 ry = _mm256_sub_ps(y, y0);

__m256 r2     = rx*rx + ry*ry;              // squared distance, 8 lanes
__m256 r_eta  = r2 + eta;                   // softened squared distance
__m256 r_32   = r_eta * sqrt(r_eta);        // (r² + η)^1.5 — see below

__m256 ax0 = m * (rx / r_32);               // m_j · rx / (r²+η)^1.5, per lane
__m256 ay0 = m * (ry / r_32);

result[0] = fast_sum(ax0);                  // Σⱼ over all 8 sources
result[1] = fast_sum(ay0);
```

The `(r² + η)^1.5` trick is worth spelling out: `r_eta` already *is*
`(distance² + η)`, so raising it to the 1.5 power is just
`r_eta × sqrt(r_eta)` — one `sqrt` and one multiply, no `pow()` call needed
in the hot path.

Because `x`, `y`, and `m` hold *all 8 bodies at once*, this single function
call computes the target's total acceleration from every source
simultaneously — including the target pulling on itself, which
contributes exactly 0 (its `rx`/`ry` are 0) and costs nothing extra to skip
explicitly.

### `get_instantenout_position`: advancing the whole system one step

```c
for (int i = 0; i < 8; i++)
    get_instantenout_acceletation(X[i], Y[i], x, y, m, eta, buffer); // per-body, scalar loop
```

This part is *not* vectorized across bodies — it's a plain loop calling the
vectorized-inner-sum function once per target. The remaining update
(`v += a·dt`, `x += v·dt`) *is* vectorized across all 8 bodies at once,
since by this point `ax_inst`/`ay_inst` are full `__m256` vectors:

```c
vx = vx + ax_inst_avx * delta_t;
vy = vy + ay_inst_avx * delta_t;
x  = x  + vx * delta_t;
y  = y  + vy * delta_t;
```

That's semi-implicit (symplectic) Euler: velocity is updated first, then
position is updated using the *new* velocity. It's a one-line change from
naive Euler but conserves energy far better over long runs, which matters a
lot once you're watching an orbit for more than a few seconds.

## Body slots & rendering

Bodies are plain parallel arrays of 8 floats — `X`, `Y`, `Vx`, `Vy`,
`masses` — index 0 is treated as the sun. `render_points` and
`render_trails` both skip any index with `mass == 0`, so an "empty" slot is
just a body with zero mass sitting somewhere off-screen; it still gets
integrated every frame (for free — it's already inside the vector), it just
never gets drawn.

Each active planet's starting velocity is computed once at startup for a
closed orbit around the sun, derived by setting centripetal acceleration
equal to the softened gravitational acceleration:

```
v = factor · r · sqrt(SUN_MASS) / (r² + η)^0.75
```

`factor = 1.0` gives a circular orbit; the `eccentricity_factor` array
scales this up or down per-planet (bounded well under the ~1.41× escape
threshold) to get visibly elliptical paths instead of identical circles.

Physics coordinates are converted to screen pixels with a simple
`center + position * scale` transform; `render_trails` keeps a fixed-size
ring buffer (`TRAIL_LEN`) of recent positions per body and fades older
points out, so orbits leave a visible tail.

## Build & run

Requires [raylib](https://www.raylib.com/) installed (a full raylib source
checkout is vendored under `raylib/` for reference, but the build links
against the system library) and a CPU with AVX2 — the Makefile compiles
with `-march=native`, so **the resulting binary is tied to the machine it
was built on** and won't run (or won't build usefully) on a CPU without the
same instruction set.

```sh
make        # build
make run    # build + run
make clean  # remove the binary
```

## Lessons learned

A few things that weren't obvious going in, surfaced by actually poking at
this simulation rather than just reading the code:

- **Mass never gates a body's own motion.** Setting a body's mass to 0 does
  not freeze it — it still falls exactly like every other body. It just
  stops *contributing* to anyone else's gravity (and, in this codebase,
  stops being drawn — see above). This confused me until I re-derived
  `a = F/m` and watched the target mass cancel out.
- **Softening isn't scale-free.** Trying to give a planet a moon exposed
  this directly: with `η = 2`, any orbit tighter than roughly `√η` units
  sits in the softened/harmonic regime rather than true inverse-square
  gravity. A moon parked close to a planet at the sun's orbital scale gets
  torn away by the sun's tidal pull almost immediately, no matter how
  massive you make the planet — and pushing the planet's mass up far enough
  to compensate destabilizes the planet's *own* orbit around the sun
  instead (its initial velocity is computed assuming a stationary sun,
  which stops being a good approximation once the planet's mass is a
  non-trivial fraction of the sun's). Moving the "moon-holding" planet much
  further from the sun — weakening the sun's tidal term, which falls off as
  `1/R³` — turned out to be the fix, not adding more mass.
- **A fixed timestep couples every orbit's timescale together.** A tight,
  fast sub-orbit (like the attempted moon) needs a much smaller `dt` to
  integrate accurately than a wide, slow one. Using one global `dt` for
  both meant integration error from the fast orbit visibly leaked into and
  distorted the slow one over time — a concrete illustration of why
  real N-body integrators use adaptive or hierarchical timestepping.
