# Episode annotation

For a short first-run walkthrough, see [TUTORIAL.md](TUTORIAL.md).

This standalone Python package segments a robot-manipulation episode into semantic
stages, optionally estimates visual hand-object contact at selected frames, and
renders review images. Its data does not feed the Upper trajectory editor.

## Install and run

Use Python 3.11 or newer. From `robotic_grounding`:

```bash
python -m pip install -r episode_annotation/requirements.txt
export OPENAI_API_KEY='your API key'
python -m episode_annotation analyze /absolute/path/episode.mp4 \
  --output-dir /absolute/path/episode_annotation_output \
  --model gpt-6.1-sol --granularity mid --contact on \
  --coarse-fps 2 --refinement-fps 8 --context-seconds 1
```

`--model` accepts `gpt-6-astra` or another Responses model that supports image input
and Structured Outputs. The tool never switches models automatically. Contact can
be disabled with `--contact off`; no contact prompt or request is then sent. Stage
granularity, sampling rates, and contact labeling are independent settings.

For an RGB frame directory, provide explicit timing:

```bash
python -m episode_annotation analyze /absolute/path/rgb_frames \
  --fps 30 --output-dir /absolute/path/frame_annotations
```

Alternatively, use `--timestamps times.json` with either an ordered list of seconds
or a filename-to-seconds mapping. A CSV needs `timestamp` and optionally `filename`
columns. Images are sorted by natural filename order. Source frame IDs are zero-based
indexes in that order; video IDs are zero-based decoded-frame indexes. Timestamps must
increase strictly. Video timestamps use decoder times when valid and FPS otherwise;
an explicit `--fps` uses that rate for all video timestamps.
The exclusive time endpoint after the last frame is estimated from FPS or the final
timestamp interval and marked as estimated in the JSON.

To regenerate images without an API call:

```bash
python -m episode_annotation render /absolute/path/episode_annotation_output/annotations.json
```

The source episode must still be present and unchanged. Output includes
`annotations.json`, `stage_contact_sheet.png` (and numbered additional pages),
`stage_timeline.png`, and `key_frames/*.png`. The JSON contains the input identity,
all source timestamps, requested and actual model IDs, sampling settings, prompt
text, API usage when supplied, objects, stages, uncertainty, and key-frame contact
estimates. `cache/` contains extracted frames and API responses keyed by source
content and analysis configuration. It may contain source images and model results;
handle it accordingly. No API key is written to this cache. Temporary HTTP
429/5xx responses are retried with bounded backoff; `Retry-After` is honored
when supplied. Quota and spend-limit errors are not retried.

Stages use half-open source-frame intervals `[start_frame, end_frame)`, with no gaps
or overlaps. Boundary estimates are distinct from inspected before/after evidence
frame IDs. Contact labels are visual estimates: `1` means estimated physical
contact, `0` means estimated absence, and `null` means uncertainty. The images
display `null` as `?`. No forces, finger contact, or grasp-quality values are inferred.

## Offline verification

From `robotic_grounding`:

```bash
python -m unittest discover -s episode_annotation/tests -v
```

This checks frame ordering, video and image timestamp mapping, endpoint sampling,
interval coverage, contact values, rendering, and the pipeline with mocked model
responses. A model-backed run requires an episode and `OPENAI_API_KEY` in the
environment.
