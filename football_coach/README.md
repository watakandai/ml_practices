# Football Positioning Coach

A policy that looks at a top-down view of a match and tells each player what to do next:
which direction to move, how fast, or where to go. The recipe mirrors robot VLAs: pretrain a
backbone that understands football, fine-tune it on good play, then improve it with RL in
[Google Research Football](https://github.com/google-research/football) (GRF). The first
version covers build-up play for the ten outfield players of the team in possession.

It implements the project plan *Football Positioning Coach — Toy Project Plan*. Milestones
and status:

| Milestone | Code | Status |
|---|---|---|
| M1 Data and alignment | `fpcoach/load_pff.py`, `labels.py`, `samples.py`, `minimap.py`, `masking.py`, `prepare_data.py`, `render.py` | Code and tests done; not yet run on the full PFF release. The GRF replay check moves to M2, since it needs GRF |
| M2 GRF setup | | Not started |
| M3 Pretraining | | Not started |
| M4 Value model and SFT | | Not started |
| M5 Coaching pictures | | Not started |
| M6 RL in GRF | | Not started |
| M7 Defense and video | | Not started |

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
cd football_coach
uv sync          # numpy, pillow, kloppy; torch is added at M3
uv run pytest    # a few seconds; uses a scripted build-up and kloppy's PFF sample files
```

Run every command from `football_coach/`, prefixed with `uv run`. Data goes to `data/` and
outputs to `runs/`; both are git-ignored.

**Real data.** PFF FC's 2022 World Cup tracking release (access terms still to confirm). The
loader expects PFF's layout: `<root>/metadata/<id>.json`, `<root>/rosters/<id>.json` and
`<root>/tracking/<id>.jsonl.bz2`.

## Pipeline (M1)

```bash
# Scripted build-up, no data needed: writes data/synthetic/{synthetic.match.npz,synthetic.samples.npz,summary.json}
python -m fpcoach.prepare_data --synthetic --out data/synthetic

# PFF World Cup: every game under the root, or a few by id
python -m fpcoach.prepare_data --pff-root /path/to/pff --out data/pff
python -m fpcoach.prepare_data --pff-root /path/to/pff --games 10517 --out data/pff

# Check labels by eye: one build-up as a GIF, with direction arrows, sprint outlines and pass/shot labels
python -m fpcoach.render --match data/pff/10517.match.npz --build-up 0 --out runs/10517_bu0.gif
```

`summary.json` lists, per game, the train/test split, the number of build-ups and how many reach
the final third, pass and shot counts, and the histogram of outfield action labels. It is the
first thing to look at on real data: the label histogram should be mostly movement directions,
with idle and sprint toggles in plausible proportions.

## Design notes

**Shared format** (`fpcoach/grf.py`, `fpcoach/match.py`). Everything is in GRF's frame: x in
[−1, 1] goal line to goal line, y in [−0.42, 0.42] with +y towards the bottom touchline, 10 steps
per second. Real pitches (about 105 × 68 m) are scaled per axis, so GRF units are anisotropic;
speeds, angles and distances are always measured in metres through `PitchScale`. A `Match`
keeps every rostered player in a fixed slot (NaN while off the pitch) with the home team
attacking +x in both periods. Velocities are central differences in GRF's "direction" units
(displacement per step) and never cross a period boundary.

**Attacking frame.** Samples are taken in the frame where the team in possession attacks +x,
like GRF's left team. The switch is a 180° rotation, not a mirror, and it rotates positions,
velocities, headings and the movement labels together (`ROTATE_180`). The test
`test_samples_are_orientation_invariant` checks that a build-up and its mirrored copy, with
the other team in possession attacking −x, give identical samples.

**State.** Per player s = (x, y, θ, vx, vy) plus goalkeeper flag and has-ball; ball
(x, y, z, vx, vy). θ is the direction of motion (tracking data has no body orientation), carried
forward when a player slows below 0.5 m/s, and set to face the opponents' goal before the
player has moved. Each team has 11 slots, goalkeeper first; slots are hidden after a red card.

**Action labels** (`fpcoach/labels.py`). One of GRF's 19 default actions per player and frame,
following GRF's sticky semantics so that a replay in GRF reproduces the movement:

- Movement: the nearest of 8 directions of the velocity, idle below 1 m/s. A direction is
  repeated while it holds, and `release_direction` is issued when a player stops.
- Sprint: on above 7 m/s, off below 5.5 m/s (hysteresis), emitted as `sprint` / `release_sprint`
  toggles. GRF takes one action per step, so a toggle delays a direction change by one step.
- Ball actions: the carrier is the nearest player within 1.5 m of a ball below 1.5 m. When the
  ball leaves a carrier for another player, the release frame is labelled. Releases within
  35 m of goal whose initial path crosses the goal line between the posts (± 2 m) are shots.
  Others are passes if the ball moved at least 3 m: high if it rose above 2 m, long if it
  travelled 30 m or more, else short. Intercepted passes count; short changes to an opponent
  are duels and get no label.
- Not labelled: GRF's dribble mode and sliding, which have no clear signature in tracking
  data. PFF's event files could replace the ball-action heuristics later.

**Build-ups** (`fpcoach/buildup.py`). Possessions (gaps of up to 0.5 s bridged) whose first
frame has the ball in the team's own third and that last at least 2 s. A build-up runs until the
ball first crosses halfway. Its outcome labels, for the value model, look at the whole
possession: did the ball reach the final third, and was there a shot. Matches are split
train/test by a hash of the match id, so frames of one match never land on both sides.

**Samples** (`fpcoach/samples.py`). One per build-up frame: both teams' states and visibility,
the ball, has-ball, GRF action labels, each own player's position 2 s later (the target head)
and the outcome labels.

**Minimap and masking** (`fpcoach/minimap.py`, `fpcoach/masking.py`). Applied at training time
so each epoch sees new masks. The minimap reproduces GRF's super mini-map exactly: 72 × 96,
layers left team / right team / ball / active, with GRF's pixel rule (its y range is ± 1/2.25,
slightly wider than the pitch). Masking hides players at random and by a camera-like rule
(only players within 30 m of the ball), which is cruder than a broadcast camera's trapezoid but
keeps near-ball players visible and far-side players hidden.

**Known gaps.** kloppy's PFF sample files put extra-time coordinates outside the pitch, so
the loader keeps periods 1 and 2 by default until that is checked on the full release.
GRF's 10 steps per second and its ball-height units are taken from the documentation and are
to be confirmed against the engine in M2.
