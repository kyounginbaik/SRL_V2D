"""Stream a local run into W&B with reference, unassisted, and VOC videos."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent.parent


def flatten(value, prefix=''):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, prefix+str(key)+'/'))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(flatten(item, prefix+str(index)+'/'))
    elif isinstance(value, (int, float)) and math.isfinite(value):
        out[prefix[:-1]] = value
    return out


def candidate_summary(candidates):
    measured = [candidate for candidate in candidates
                if isinstance(candidate.get('cost'), (int, float))
                and math.isfinite(candidate['cost'])]
    result = dict(candidate_count=len(measured), skipped_window_count=len(candidates)-len(measured),
                  gate_pass_fraction=sum(candidate.get('gate_passed', candidate.get('accepted', False))
                                         for candidate in measured)/max(len(measured), 1))
    if measured:
        costs = [candidate['cost'] for candidate in measured]
        result.update(candidate_cost_min=min(costs), candidate_cost_mean=sum(costs)/len(costs))
    return result


def feedback_summary(records):
    """Keep per-branch details local while exposing the measured feedback loop."""
    fitted = [record for record in records if not record.get('trial_critic_skipped', True)]
    result = dict(branch_checkpoints=len(records), fitted_branch_checkpoints=len(fitted))
    for key in ('trial_feedback_rows', 'trial_feedback_complete_rows',
                'trial_feedback_discarded_rows', 'trial_feedback_trajectories'):
        result[key] = sum(record.get(key, 0) for record in records)
    value_keys = ('trial_critic_heldout_return_mae_before', 'trial_critic_heldout_return_mae_after',
                  'trial_critic_prediction_before', 'trial_critic_prediction_after',
                  'trial_critic_measured_return')
    keys = [key+suffix for key in value_keys for suffix in ('', '_normalized', '_raw')]
    keys += ['trial_critic_train_loss', 'trial_critic_train_loss_normalized',
             'trial_critic_cost_scale', 'trial_critic_mc_weight']
    for key in keys:
        values = [record[key] for record in fitted if key in record]
        if values:
            result[key] = sum(values)/len(values)
    result['trial_critic_parameter_delta_l2_max'] = max(
        (record.get('trial_critic_parameter_delta_l2', 0) for record in fitted), default=0)
    return result


def main():
    import cv2
    import wandb
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('smoke', 'smoke_retry', 'smoke_retry2', 'smoke_retry3', 'smoke_retry4', 'pilot', 'pilot_stepvoc', 'contact_smoke', 'contact_smoke_v2', 'contact_pilot', 'guided_smoke', 'guided_pilot', 'refit_smoke', 'refit_pilot', 'refit_control_pilot', 'surface_smoke', 'surface_pilot', 'local_smoke', 'local_pilot', 'main', 'ablation_4', 'ablation_16', 'ablation_1024'))
    parser.add_argument('episode')
    parser.add_argument('variant', choices=('fixed', 'chunk'))
    parser.add_argument('seed', type=int)
    parser.add_argument('--once', action='store_true', help='Backfill current files and exit')
    args = parser.parse_args()
    root = BASE/'runs'/args.phase/args.episode/args.variant/f'seed_{args.seed}'
    train = root/'training'
    config_path = root/'run_config.json'
    while not config_path.exists():
        if args.once:
            raise FileNotFoundError(config_path)
        time.sleep(30)
    work = root/'wandb_sync'
    work.mkdir(parents=True, exist_ok=True)
    sync_lock = (work/'sync.lock').open('a')
    try:
        fcntl.flock(sync_lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.once else 0))
    except BlockingIOError:
        print('Live W&B sync owns this run; it will upload the completed checkpoint.', flush=True)
        return
    state_file = work/'cursor.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {'offsets':{}, 'videos':{}}
    # Older syncs logged media without environment_steps, so W&B displayed a
    # historical VOC-on clip beside the run's current VOC=0 metric. Re-upload
    # completed clips once with their checkpoint step as the media x-axis.
    if state.get('media_schema_version', 0) < 2:
        state['video_paths'] = {}
        state['videos'] = {}
        state['media_schema_version'] = 2
    run_id = f'{os.environ.get("UNIRETARGET_RUN_TAG", "chord100m")}-{args.phase}-{args.episode}-{args.variant}-s{args.seed}'
    config = json.loads(config_path.read_text())
    hypergradient = config.get('engine') == 'wrench_hypergradient'
    experiment = config.get('experiment_name', 'wrench-hypergradient' if hypergradient else 'chord-fixed-start')
    experiment_date = config.get('experiment_date', '20261015')
    name = (f'{experiment}-full-d56-w{config["worlds"]}-{args.phase}-{args.episode}-s{args.seed}'
            if hypergradient else
            f'chord-fixed-start-{config.get("upper_optimizer", "gaussian")}-d56-c{config["upper_trials"]}-{args.phase}-{args.episode}-s{args.seed}')
    run = wandb.init(entity=os.environ.get('WANDB_ENTITY', 'DexDreamer'), project=os.environ.get('WANDB_PROJECT', 'v2d-track3-uniretarget'),
                     group=f'{experiment}-{experiment_date}-{args.phase}', id=run_id,
                     name=name,
                     resume='allow', dir=str(work), config=config)
    run.define_metric('environment_steps')
    run.define_metric('*', step_metric='environment_steps')
    for scope in ('track3_estimated', 'track3_assisted_estimated'):
        for metric in ('auc', 'sp_sr', 'mp_sr', 'mppe_cm', 'rpe_cm', 'cd_o_cm'):
            run.define_metric(f'{scope}/{metric}', step_metric='environment_steps')
    run.summary['track3_estimates_notice'] = 'Reference poses and meshes are proxy GT; CD-O is mesh self-comparison; main panel is VOC=0.'
    if hypergradient:
        run.summary['upper_cost_units'] = (
            f'Critic normalized return = {config.get("upper_cost_scale", 1.)} * raw dimensionless wrench-support return. '
            'Explicit *_raw and *_normalized metrics distinguish them; physical acceptance cost has different terms.')
    run.define_metric('voc_schedule/*', hidden=True)
    # Explicit definitions also hide previously logged metadata on run pages;
    # a namespace glob does not reliably update existing metric definitions.
    for label in ('physics_checkpoint', 'assisted_voc', 'reference_overlay'):
        for field in ('checkpoint_step', 'applied_voc_scale'):
            key = f'video_context/{label}/{field}'
            run.define_metric(key, hidden=True, summary='none', overwrite=True)
            try:
                del run.summary[key]
            except KeyError:
                pass
    (work/'run.json').write_text(json.dumps({'url':run.url,'id':run.id},indent=2)+'\n')

    def stream(filename, prefix):
        path = train/filename
        if not path.exists():
            return
        with path.open() as source:
            source.seek(state['offsets'].get(filename, 0))
            while True:
                line = source.readline()
                if not line or not line.endswith('\n'):
                    break
                record = json.loads(line)
                step = record.get('step')
                if step is None:
                    raise ValueError(f'Missing step in {path}')
                if prefix == 'upper/' and 'candidates' in record:
                    candidates = record.pop('candidates')
                    record.update(candidate_summary(candidates))
                    # Full per-candidate evidence stays in search_batches/*/scores.json.
                if prefix == 'upper/':
                    feedback = record.pop('critic_feedback', [])
                    if feedback:
                        run.log({'environment_steps': int(step),
                                 **flatten(feedback_summary(feedback), 'upper_feedback/')})
                    failure = record.pop('failure_context', {})
                    record['failure_time_seconds'] = failure.get('failure_seconds')
                    record['error_onset_seconds'] = failure.get('onset_seconds')
                    for key in ('consensus_fraction', 'consensus_reached', 'baseline_failure_fraction',
                                'baseline_probe_worlds', 'baseline_probe_seconds', 'selected_object_index'):
                        if key in failure:
                            record[key] = failure[key]
                    record['windows_tested'] = len(record.pop('windows', []))
                    record.pop('window_results', None)
                    attempts = record.pop('confirmation_attempts', [])
                    record['confirmation_attempt_count'] = len(attempts)
                    robust = record.pop('robust_confirmation', {})
                    checks = robust.get('checks', [])
                    if checks:
                        record['confirmation_passed'] = bool(robust.get('passed'))
                        record['confirmation_cost_gain_mean'] = sum(c['cost_gain'] for c in checks)/len(checks)
                        record['confirmation_improving_fraction'] = sum(c['cost_gain']>0 for c in checks)/len(checks)
                    # Detailed paired trajectories/metrics remain in local scores.json.
                metric_prefix = prefix
                if prefix == 'adaptation/':
                    # Publish progress while long validation is running, without
                    # creating a separate chart for every reference frame.
                    metric_prefix += record.pop('branch')+'/'
                    record = dict(step=step, environment_transitions=record['environment_steps'],
                                  cost=record['cost'], training={key: value for key, value in record['training'].items()
                                                               if key != 'frame_samples'})
                metrics = {k:v for k,v in flatten(record, metric_prefix).items() if not k.startswith('voc_schedule/')}
                metrics['environment_steps'] = int(step)
                run.log(metrics)
                state['offsets'][filename] = source.tell()

    completed_since = None
    try:
        while True:
            stream('training_metrics.jsonl', '')
            stream('upper_updates.jsonl', 'upper/')
            if hypergradient:
                stream('adaptation_progress.jsonl', 'adaptation/')
            stream('progress_checks.jsonl', 'progress/')
            if config.get('voc_mode') != 'step_schedule':
                stream('voc_gate.jsonl', 'voc_gate/')
            metrics_path = train/'training_metrics.jsonl'
            metric_lines = metrics_path.read_text().splitlines(keepends=True) if metrics_path.exists() else []
            metric_lines = [line for line in metric_lines if line.endswith('\n')]
            latest_training = json.loads(metric_lines[-1]) if metric_lines else {}
            current_step = int(latest_training.get('step', 0))
            current_voc = latest_training.get('curriculum/voc_scale')
            version_steps = {0: 0}
            updates_path = train/'upper_updates.jsonl'
            if updates_path.exists():
                for line in updates_path.read_text().splitlines(keepends=True):
                    if not line.endswith('\n'):
                        continue
                    update = json.loads(line)
                    if update.get('accepted'):
                        version_steps[int(update['version_after'])] = int(update['step'])
            for label, paths in (
                ('reference_overlay', (p for p in (root/'reference_overlays').glob('version_*/transparent_closeups/overlay.mp4') if (p.parent/'complete.json').exists())),
                ('physics_checkpoint', (p for p in (root/'checkpoint_videos').glob(f'step_*/{args.episode}.mp4') if p.with_suffix('.complete').exists())),
                ('assisted_voc', (p for p in (root/'voc_checkpoint_videos').glob(f'step_*/{args.episode}.mp4') if p.with_suffix('.complete').exists())),
            ):
                candidates = sorted(paths, key=(
                    (lambda path: int(path.parent.name.removeprefix('step_')))
                    if label != 'reference_overlay' else
                    (lambda path: int(path.parent.parent.name.removeprefix('version_')))
                ))
                for path in candidates:
                    uploaded = state['video_paths'].setdefault(label, [])
                    if str(path) in uploaded:
                        continue
                    capture = cv2.VideoCapture(str(path))
                    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0,count-1))
                    good, _ = capture.read()
                    capture.release()
                    expected = next(x['frames'] for x in json.loads((BASE/'frozen_manifest.json').read_text()) if x['episode']==args.episode)
                    if count != expected or not good:
                        raise ValueError(f'Invalid video {path}: {count} frames, expected {expected}')
                    if label == 'reference_overlay':
                        version = int(path.parent.parent.name.removeprefix('version_'))
                        video_step = version_steps.get(version, current_step)
                        caption = f'OVERLAP + HANDLE CLOSE-UPS: cyan original hands (45% opacity) + solid current v{version} | same camera and true scale | step {video_step:,}'
                        if version == 0: caption += ' | no accepted refinement yet'
                        scale = None
                    else:
                        manifest = json.loads((path.parent/'manifest.json').read_text())
                        video_step = int(manifest['environment_steps'])
                        scale = float(manifest['voc_scale'])
                        caption = f'checkpoint step {video_step:,} | VOC {scale:g} | full-horizon physics'
                        if label == 'assisted_voc':
                            caption += ' | current checkpoint training VOC setting'
                    payload = {'environment_steps': video_step,
                               f'videos/{label}': wandb.Video(str(path), format='mp4', caption=caption)}
                    run.log(payload)
                    state['videos'][label] = str(path)
                    uploaded.append(str(path))
            # Separate VOC=0 estimates from assisted diagnostics. Persist a cursor
            # per checkpoint so reconnects cannot silently lose historical scores.
            for folder, namespace in (('checkpoint_videos', 'track3_estimated'),
                                      ('voc_checkpoint_videos', 'track3_assisted_estimated')):
                for report_path in sorted((root/folder).glob('step_*/track3_estimates.json'),
                                          key=lambda p: int(p.parent.name[5:])):
                    uploaded_metrics = state.setdefault('track3_reports', [])
                    if str(report_path) in uploaded_metrics:
                        continue
                    report = json.loads(report_path.read_text())
                    if namespace == 'track3_estimated' and report['voc_scale'] != 0:
                        raise ValueError('Assisted rollout in unassisted Track 3 panel')
                    metrics = {f'{namespace}/{key}': value for key,value in report['metrics'].items()}
                    run.log({'environment_steps': int(report_path.parent.name[5:]), **metrics})
                    run.summary.update(metrics)
                    uploaded_metrics.append(str(report_path))
            run.summary['latest_training_environment_steps'] = current_step
            if current_voc is not None:
                run.summary['latest_training_voc_scale'] = float(current_voc)
            run.summary['environment_steps'] = current_step
            stage_file = root/'stage.json'
            stage = json.loads(stage_file.read_text()) if stage_file.exists() else {'stage':'pending'}
            run.summary['pipeline_stage'] = stage['stage']
            metrics = root/'evaluation'/'metrics.json'
            if metrics.exists() and not state.get('evaluation'):
                run.summary.update(flatten(json.loads(metrics.read_text()),'evaluation/'))
                state['evaluation'] = True
            state_file.write_text(json.dumps(state,indent=2)+'\n')
            if args.once:
                break
            if stage['stage'] in ('complete','failed','cancelled_by_user'):
                break
            time.sleep(30)
    finally:
        terminal_stage = json.loads((root/"stage.json").read_text()).get("stage") if (root/"stage.json").exists() else None
        run.finish(exit_code=1 if terminal_stage == "failed" else 0)


if __name__ == '__main__':
    main()
