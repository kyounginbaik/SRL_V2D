# Episode annotation quickstart

This standalone tool segments one manipulation video into stages and can estimate
hand–object contact at selected keyframes. It runs on the host with Python 3.11+
and does not require the robotics training container. See [README.md](README.md)
for all input formats and options.

## Run one episode

From `robotic_grounding/` in the V2D repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r episode_annotation/requirements.txt

# Set OPENAI_API_KEY in your shell before running; do not put the key in Git.
python -m episode_annotation analyze /absolute/path/to/episode_000000.mp4 \
  --output-dir /absolute/path/to/annotation_output \
  --model gpt-6.1-sol --granularity mid --contact on
```

Use a model that supports image input and Structured Outputs. The command makes
paid API calls. The `episode_000000` reference run used 60 calls and had an
estimated cost of about $1.88; cost and runtime vary by video and settings.
Run one episode and inspect it before processing more. Use `--contact off` if
you only need stages; it skips contact requests.

## Inspect the result

Open these files in the output directory:

| File | What to look for |
| --- | --- |
| `stage_timeline.png` | Stage names and time intervals. |
| `stage_contact_sheet.png` and numbered pages | Selected keyframes with stage and contact labels. |
| `key_frames/` | Individual annotated panels. |
| `annotations.json` | Frame boundaries, objects, hand roles, contact evidence, and usage. |

Each contact entry labels one **anatomical hand–object pair at one source frame**.
`1` means estimated touch, `0` means estimated absence, and `null` (shown as
`?` in images) means uncertain. Only pairs selected for a stage are checked;
these are visual estimates, not continuous measurements or force labels.
Compare uncertain labels and short stage transitions with the source video.

To regenerate the images from an existing `annotations.json` without API calls:

```bash
python -m episode_annotation render /absolute/path/to/annotation_output/annotations.json
```

The original video must still be available and unchanged. Videos, generated
outputs, and the `cache/` directory are local data; arrange access to them
separately from the Git branch.
