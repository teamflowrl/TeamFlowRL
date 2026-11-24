#!/usr/bin/env python3
"""Reproduce the frozen TCP-distance threshold calibration of the hands-away metric (CPU only, no dependencies)."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def summarize(values):
    values = sorted(x for x in values if x is not None)
    def q(p):
        if not values:
            return None
        i = (len(values)-1)*p
        return values[math.floor(i)]*(1-(i % 1))+values[math.ceil(i)]*(i % 1)
    return dict(n=len(values), p50=q(.5), p90=q(.9), p95=q(.95), maximum=max(values, default=None))


def episode_stats(data):
    rows = data['trajectory']
    assert len(rows) == 150 and len(data['episodes']) == 100
    assert all(rows[j]['step'] == rows[j-1]['step']+1 for j in range(1,len(rows)))
    episodes = []
    for i, ep in enumerate(data['episodes']):
        window, scores, pointwise, arms = [], [], [], [[], []]
        for r in rows:
            dist = r['tcp_cube_distance_m'][i]
            assert len(dist) == 2 and all(math.isfinite(x) and x >= 0 for x in dist)
            if r['original'][i] and r['placed'][i]:
                pointwise.append(min(dist))
                for a in range(2):
                    arms[a].append(dist[a])
                window.append(min(dist))
                if len(window) > 10:
                    window.pop(0)
                if len(window) == 10:
                    scores.append(min(window))
            else:
                window = []
        episodes.append(dict(seed=ep['seed'], distance_10step_m=max(scores, default=None),
                             valid_10step_windows=len(scores), valid_original_steps=len(pointwise),
                             distance_pointwise_max_m=max(pointwise, default=None),
                             arm_distance_original_max_m=[max(x, default=None) for x in arms]))
    episodes.sort(key=lambda x:x['seed'])
    assert len({x['seed'] for x in episodes}) == 100
    for j, ep in enumerate(episodes):
        ep['split'] = 'calibration' if j < 50 else 'validation'
    return episodes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, default=Path(__file__).resolve().parents[1]/'results/baseline')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((Path(__file__).resolve().parent/'threshold_protocol.json').read_text())
    evidence=dict(protocol=protocol, protocol_sha256=hashlib.sha256((Path(__file__).resolve().parent/'threshold_protocol.json').read_bytes()).hexdigest(), runs={})
    for task in ('ms','rf'):
        for method in ('teamflow','dfpo','maogpo','fppo'):
            name=f'{task}_{method}'; path=args.baseline_dir/name/'baseline.json'; data=json.loads(path.read_text())
            assert data['stochastic'] is False and data['phase']=='baseline'
            eps=episode_stats(data)
            evidence['runs'][name]=dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_sha256=data['checkpoint_sha256'], checkpoint=data['checkpoint'], source_step=data['source_step'],
                episodes=eps, distributions={split:summarize(e['distance_10step_m'] for e in eps if e['split']==split) for split in ('calibration','validation')})
    thresholds={}; accepted=True
    for task,floor in [('ms',.12),('rf',.15)]:
        runs={k:v for k,v in evidence['runs'].items() if k.startswith(task+'_')}
        max_cal=max(e['distance_10step_m'] for r in runs.values() for e in r['episodes'] if e['split']=='calibration' and e['distance_10step_m'] is not None)
        threshold=max(floor, max_cal+.01)
        thresholds[task]=dict(tcp_cube_distance_m=threshold,contact_force_n=.001,opening_m=.06,hold_steps=10,
                              linear_speed_m_s=None if task=='ms' else .02, angular_speed_rad_s=None if task=='ms' else .2)
        for r in runs.values():
            r['threshold_m']=threshold
            r['validation']={}
            for split in ('calibration','validation'):
                eps=[e for e in r['episodes'] if e['split']==split]
                valid=[e for e in eps if e['distance_10step_m'] is not None]
                false=[e for e in valid if e['distance_10step_m'] >= threshold]
                sufficient=len(valid)>=protocol['minimum_valid_episodes_per_split_per_method']
                rate=len(false)/len(valid) if valid else None
                passed=sufficient and rate<=.05 and len(false)/len(eps)<=.05
                r['validation'][split]=dict(total_episodes=len(eps), valid_episodes=len(valid), false_positive_episodes=len(false),
                    false_positive_seeds=[e['seed'] for e in false], rate_all=len(false)/len(eps), rate_conditional=rate,
                    passed=passed, zero_failure_one_sided_95pct_upper_conditional=1-.05**(1/len(valid)) if valid and not false else None)
                accepted=accepted and passed
    evidence['accepted']=accepted
    dump(out/'threshold_evidence.json',evidence)
    dump(out/'thresholds.json',thresholds)
    lines=['# TCP-to-cube distance thresholds','',f'Held-out validation: {"passed" if accepted else "failed (do not start training)"}. Offline recomputation from baseline trajectories.','',
           '## Fixed rule','',
           'See `threshold_protocol.json`. 100 episodes per method; seeds sorted ascending, first 50 for calibration and last 50 for validation. Only 10-step windows in which the original and placed flags both hold are considered. Per window, take the minimum TCP-to-nearest-cube distance over both arms and 10 steps; per episode, take the maximum over windows. Episodes without a valid window never count as withdrawal.','',
           'Per-task threshold = max(fixed floor, largest calibration statistic across the four methods + 0.01 m). Floors: MS 0.12 m, RF 0.15 m. Validation requires an all-episode and valid-episode false-positive rate of at most 5% per method, with at least 20 valid episodes per split.','',
           '| task | TCP distance threshold (m) |','|---|---:|']
    lines += [f'| {k} | {v["tcp_cube_distance_m"]:.12f} |' for k,v in thresholds.items()]
    lines += ['', '| run | calibration valid/50 | calibration max (m) | validation valid/50 | false/valid | conditional false-positive rate | evidence |', '|---|---:|---:|---:|---:|---:|---|']
    for name,r in evidence['runs'].items():
        c=r['validation']['calibration']; v=r['validation']['validation']
        lines.append(f'| {name} | {c["valid_episodes"]}/50 | {r["distributions"]["calibration"]["maximum"]:.8f} | {v["valid_episodes"]}/50 | {v["false_positive_episodes"]}/{v["valid_episodes"]} | {v["rate_conditional"]:.2%} | `threshold_evidence.json:runs.{name}` |')
    lines += ['', 'Recompute with `python3 calibrate_thresholds.py --baseline-dir <baseline results>`. No GPU is needed.']
    (out/'THRESHOLDS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(accepted=accepted,thresholds=thresholds,validation={k:v['validation'] for k,v in evidence['runs'].items()}),indent=2))
    if not accepted:
        raise SystemExit('Held-out validation failed or insufficient: do not train; obtain fresh evaluation data under a new frozen protocol.')

if __name__=='__main__':main()
