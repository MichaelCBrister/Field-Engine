# Field-Theoretic Redesign: From 5 Heuristics to 3 Field Observables

## Part 1: What Each Term Actually Computes

### W_MATERIAL (current value: 2.986)

The material term reads `b.material`, which is the running sum of all signed
piece charges on the board (White pieces positive, Black pieces negative). The
piece charges are fixed constants in state.jl: pawn=1, knight=3, bishop=3.25,
rook=5, queen=9, king=100. Because both kings are always present, their +100
and -100 cancel, so `b.material` equals the classical material balance in
pawn-equivalent units. A position where White has an extra knight reads
material=+3.0.

This term has **no connection to the field model**. It is a direct inventory
count of charges, analogous to measuring total electric charge in a volume
without consulting the field at all. In electrostatics, total charge determines
the *far-field* behavior (Gauss's law: the flux through any enclosing surface
equals the enclosed charge). The field *already encodes* material — a position
with more White pieces will have a more positive field integral. The reason
W_MATERIAL exists as a separate term is that `sum(field)` is a noisy proxy for
material: the field sum depends on piece *positions* (a rook in the corner has
different field sum than a centralized rook), so the optimizer needs a clean,
position-independent material signal. But this is a workaround for the field
model not being expressive enough, not a fundamental necessity.

### W_FIELD (current value: -0.096)

The field control term computes `sum(field)`, the integral of the potential
over all 64 squares. Each piece at position (r,f) with charge Q radiates
Phi(r',f') = Q / (1 + d^2) where d is the piece-type-specific distance
(ray-following for sliders, L-jump for knights, Chebyshev for king, directional
for pawns). The total field is the superposition (sum) of all individual piece
fields. Summing this 8x8 matrix yields a scalar that measures net
"territorial influence" — how much of the board is under White vs Black
control, weighted by proximity to the controlling pieces.

The small, *negative* learned weight (-0.096) is revealing. It says the
optimizer found that having a large positive field sum is actually slightly
*bad* for White. This is counterintuitive until you realize that `sum(field)`
is dominated by the king charges (+/-100). A White king that is surrounded by
empty space radiates a large positive field — but an exposed king is a
*liability*. The negative weight is the optimizer's way of encoding "don't let
your king's field dominate the board unchallenged" — which is really a king
safety signal in disguise. The meaningful spatial information (piece
centralization, space advantage) is buried under the king's overwhelming
contribution. This is a sign that the field sum is too crude an observable:
it conflates positional control with king exposure.

### W_KING_SAFETY (current value: 0.438)

King safety computes `king_zone_pressure(field, enemy_king) -
king_zone_pressure(field, own_king)`, where `king_zone_pressure` sums
`max(0, -field[r,f] * color)` over the 8 squares adjacent to a king. This
extracts only the *enemy-signed* field contributions near each king. For White's
king, it sums the negative (Black-controlled) field values in the 3x3 zone; for
Black's king, it sums the positive (White-controlled) values. The difference
measures "how much more are we pressuring their king than they're pressuring ours."

In field theory terms, this is a **localized flux measurement**: the integral of
the opponent's field through a surface (the king zone). It is physically
meaningful — it corresponds to the force that enemy "charges" (pieces) exert in
the king's neighborhood. However, it is *manually localized*. The field itself
contains this information globally; the 3x3 window is a hand-coded kernel that
the engine imposes. A richer field model would not need this extraction step —
the field's local behavior near kings would naturally contribute to the global
energy through a properly designed energy functional.

### W_TENSION (current value: -0.000, effectively zero)

Tension computes the squared gradient magnitude of the field near each king:
sum of (field[r,f] - field[r+dr, f+df])^2 over all adjacent pairs in each
king's 3x3 zone. High tension means the field changes rapidly — White and Black
influence are colliding. This is the discrete analogue of |nabla Phi|^2, which
in electrostatics equals the local energy density of the field.

CMA-ES drove this weight to zero. This is the strongest possible signal that
the term is redundant given the other four. And physically, it *should* be
redundant: tension near the king zone is a second-order consequence of enemy
pressure. If the opponent has pieces projecting field into our king zone
(high king_safety), then the field must also have steep gradients there (high
tension). The two observables are correlated to the point of collinearity. With
only 5 dimensions and noisy fitness, CMA-ES correctly decided that adding
tension on top of king_safety provides no extra signal — its weight should be
zero.

### W_MOBILITY (current value: 1.285)

Mobility computes the total number of squares reachable by each side's pieces,
differenced (White mobility - Black mobility). This is calculated by
`compute_mobility_count`, which iterates over all pieces, walks their legal
move rays (stopping at blockers and own-color pieces), and sums the counts.

This term is **entirely outside the field model**. It does not read the field
matrix at all. It is a pure move-counting heuristic borrowed from conventional
chess engines. In field-theoretic terms, mobility is closest to "flux" — the
total field flowing outward from your pieces — but it is computed independently
from the actual field values. The field model already captures piece activity
implicitly (an unblocked rook radiates field along the entire file, while a
blocked rook's field is clipped), but `sum(field)` doesn't separate "my pieces
are active" from "I have more material." Mobility exists because the single
observable `sum(field)` is not rich enough to decompose these effects.


## Part 2: Emergent vs Hand-Coded

| Term | Reads field? | Localized? | Truly emergent? |
|---|---|---|---|
| W_MATERIAL | No | No | No — raw charge count, independent of field |
| W_FIELD | Yes (sum) | No | Partially — crude global integral |
| W_KING_SAFETY | Yes (3x3 zone) | Hand-coded window | No — manual spatial filter on the field |
| W_TENSION | Yes (gradient in 3x3) | Hand-coded window | No — derived observable with manual extraction |
| W_MOBILITY | No | No | No — move counting, ignores the field entirely |

**Verdict**: Only W_FIELD interacts with the field at all, and only through a
global sum that throws away spatial structure. The other four terms are either
inventories of the board state (material, mobility) or hand-coded spatial
extractions from the field (king safety, tension). None of them emerge from
the field model in the way that, say, the Hamiltonian of an electromagnetic
field produces forces on charges automatically.

The fundamental problem is that the "energy functional" is a **linear
combination of hand-picked observables**, not an integral of a local energy
density derived from the field. In physics, the energy of an electric field is:

    E = (epsilon_0 / 2) * integral |nabla Phi|^2 dV

This single expression automatically encodes all the information about charge
distribution, spatial control, tension, and inter-charge interactions. FieldEngine
instead computes Phi and then discards most of its structure, keeping only
`sum(Phi)` and a few hand-extracted statistics.


## Part 3: What Could Be Removed/Merged

**Remove immediately: W_TENSION.** CMA-ES already zeroed it. It is collinear
with W_KING_SAFETY and adds no signal. Removing it simplifies the weight
space from 5D to 4D, which directly improves CMA-ES convergence (the curse
of dimensionality hits hard below n=10, and dropping from 5 to 4 reduces the
search volume by a factor of ~3 at fixed sigma).

**Merge candidates**: W_KING_SAFETY could be absorbed into a richer W_FIELD
term if the field observable were not a flat sum but a *spatially structured*
integral that naturally weights king-adjacent squares more heavily. This is the
core insight of the proposal below.

**W_MOBILITY is the hardest to remove** because it captures information
(piece activity, ray blockage) that the current field observable does not expose.
However, the field model *already computes* ray blockage (sliding fields are
clipped by intervening pieces). The problem is that `sum(field)` doesn't
distinguish "unblocked rook with long rays" from "extra pawn of material."
A quadratic field term (see proposal) would separate these.


## Part 4: Concrete Proposal — From 5 Linear Terms to 3 Field Observables

### Current model (5 terms, linear)

    E = w1 * material + w2 * sum(Phi) + w3 * king_pressure + w4 * tension + w5 * mobility

### Proposed model (3 terms)

    E = w1 * material + w2 * sum(Phi) + w3 * sum(Phi^2)

Where `Phi^2` means the element-wise squared field: `sum(field[r,f]^2 for all r,f)`.

#### What does sum(Phi^2) compute?

The squared field integral, `integral Phi^2 dV`, is the **field energy** — a
standard observable in physics that measures how "intense" the field is
regardless of sign. In the discrete chess setting:

    sum(Phi^2) = sum_{r,f} field[r,f]^2

This single number encodes *all* of the following:

1. **King safety**: The king's charge is +/-100, producing enormous field
   values (~100, 50, 20) near itself. When enemy pieces project opposing field
   into the king zone, the net field values shrink (partial cancellation),
   which *reduces* Phi^2. Conversely, if the king is uncontested, Phi^2 stays
   high near the king. The sign of w3 determines whether the engine prefers
   high local intensity (defensive king) or low local intensity (enemy
   field has penetrated — dangerous). **This subsumes W_KING_SAFETY without
   a hand-coded 3x3 window.**

2. **Piece activity / mobility**: An unblocked rook radiates field across 14
   squares; a blocked rook radiates across 3. The unblocked rook contributes
   more to sum(Phi^2) because its field reaches more squares. A centralized
   knight contributes 8 terms of (Q*0.8)^2 = 5.76 each, while a cornered
   knight contributes only 2. **This is a continuous proxy for mobility that
   emerges from the field structure without move counting.**

3. **Tension / gradient**: Regions where White and Black fields collide have
   large opposing contributions that partially cancel in Phi but do NOT cancel
   in Phi^2 (because (-a)^2 + b^2 != (b-a)^2). High-tension zones contribute
   more to sum(Phi^2) than peaceful zones. **This subsumes W_TENSION without
   explicit gradient computation.**

4. **Coordination**: Two rooks on the same file that reinforce each other's
   field produce a superlinear contribution to sum(Phi^2) (because (a+b)^2 >
   a^2 + b^2 when a,b have the same sign). The squared term automatically
   rewards piece coordination.

#### Why keep material and sum(Phi) alongside sum(Phi^2)?

- **Material** provides a clean, position-independent anchor. Without it, the
  engine has no way to distinguish "I have an extra queen" from "my pieces
  are well-placed." Material is a topological invariant of the position (it
  only changes on captures); the field terms are geometric.

- **sum(Phi)** captures the net signed influence. It distinguishes "White
  controls the board" from "Black controls the board." sum(Phi^2) is always
  positive and cannot do this. The two are complementary: sum(Phi) is the
  first moment (mean field), sum(Phi^2) is the second moment (field energy).

#### What changes in the code?

**energy.jl / search.jl** — Replace `eval_w`:

```julia
function eval_w(b::Board, w::Vector{Float64}, field::Matrix{Float64})::Float64
    material   = b.material
    field_sum  = 0.0
    field_sq   = 0.0
    @inbounds for f in 1:8, r in 1:8
        v = field[r, f]
        field_sum += v
        field_sq  += v * v
    end
    return w[1] * material + w[2] * field_sum + w[3] * field_sq
end
```

This is **faster** than the current eval_w: one pass over the 8x8 matrix
instead of calling `sum(field)` + `king_zone_pressure` (two 3x3 scans) +
`tension_near_king` (two 3x3 scans with gradient) + `compute_mobility_count`
(full board scan with ray walking). The per-eval cost drops from ~400
operations to ~130.

**fields.jl** — No changes. The field computation is correct and well-designed.
The problem was never in how Phi is computed, but in how it is *read*.

**optimize.jl** — Change weight bounds from 5D to 3D:

```julia
const WEIGHT_MIN = Float64[2.0,  -1.5, -0.01]
const WEIGHT_MAX = Float64[20.0,  1.5,  0.01]
```

The w3 (field energy) weight will be small because sum(Phi^2) includes king
field contributions of order 10^4. CMA-ES will find the right scale.

### Why this deepens the physics

The current model treats the field as a black box and then grafts chess
heuristics on top of it. The proposed model says: the field *is* the
evaluation. We read it the way physicists read fields — through its moments
(integral Phi, integral Phi^2). King safety, tension, and mobility are not
separate concepts bolted on; they are consequences of the field structure that
emerge automatically from the second moment.

This is closer to how energy works in actual field theories:
- In electrostatics: E = (1/2) integral epsilon |nabla Phi|^2 dV
- In scalar field theory: E = integral [(nabla Phi)^2 + m^2 Phi^2] dV
- In FieldEngine (proposed): E = w1*Q_total + w2*integral(Phi) + w3*integral(Phi^2)

The first two are standard; the third is a legitimate lattice field theory on
a finite 8x8 grid.

### Expected impact on learning

1. **Faster CMA-ES convergence**: 3D search space instead of 5D. The volume
   ratio is (5D hypersphere / 3D hypersphere) ~ 3:1 at fixed sigma. Expect
   convergence in ~60% of current generations.

2. **Reduced overfitting**: Fewer parameters = less risk that CMA-ES finds
   weights that exploit noise in self-play rather than genuine position quality.

3. **Cheaper per-evaluation**: Eliminating mobility counting (the most
   expensive component of eval_w, requiring full ray-walking) and the king zone
   scans cuts per-eval cost by roughly 60%. At depth 5, this translates to
   millions fewer operations per generation.

4. **Risk**: The squared field term may not fully replace mobility. Piece
   activity is genuinely important, and sum(Phi^2) captures it only through
   the field proxy. If the field model's ray-blocking is not precise enough
   (e.g., knights' non-physical field), the proxy may be weak. Mitigation: run
   a 200-gen CMA-ES trial on the 3-term model and compare win rate against
   the 5-term model. If the 3-term model loses, add mobility back as a 4th
   term (still one fewer than today).

### Alternative considered: gradient energy term

Instead of sum(Phi^2), we could use sum(|nabla Phi|^2) — the actual
electrostatic field energy density. This would more directly capture tension
and boundary effects. However:

- It requires computing all 64 gradient values (128 subtractions + 64
  squarings + 64 additions), which is ~2x the cost of sum(Phi^2)
- It is more sensitive to the discrete grid artifacts (edge effects at board
  boundaries)
- sum(Phi^2) already captures gradient information indirectly (regions with
  steep gradients have large Phi values on at least one side)

The gradient term is worth exploring as a follow-up (making a 4-term model:
material, Phi, Phi^2, |nabla Phi|^2), but the minimal 3-term model should be
tested first.
