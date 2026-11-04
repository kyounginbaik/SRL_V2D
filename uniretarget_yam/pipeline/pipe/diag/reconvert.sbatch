#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
N=$1; EP=$(printf recon_%06d $N); ZEP=$(printf episode_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; W=/scratch2/exaFLOPs/latest_recon/run37; B=/scratch2/exaFLOPs/hawor_rerun/root; E=$H/episodes/$EP; C=/uniretarget_yam/episodes/$EP; META=$E/recon_meta.json; TMP=$H/.cache/tmp_rc_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
fl() { apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1 -B "$TMP:/tmp" -B "$H:/uniretarget_yam" -B /scratch2 --pwd $W $R/flash_chord_yam.sif "$@"; }
read FR PAD SIDE MAINOBJ <<<$(python3 -c "
import json;d=json.load(open('$META'));m=max(d['movers'],key=lambda x:x['move_cm']);print(d['frames'],d.get('pad',0),m['side'],m['object'])")
fl python $W/convert_hawor.py --frames $FR --pad $PAD --hawor $B/hands/$ZEP/hawor_${MAINOBJ}_result.npz --target $B/targets/$ZEP/${MAINOBJ}_trajectory.npz --up $E/up.json --side $SIDE --output $E/bundle | tail -1 | cut -c1-200
for X in $(cat $E/other_bundles.txt); do KK=${X%%:*}; NN=${X#*:}; fl python $W/convert_hawor.py --frames $FR --pad $PAD --hawor $B/hands/$ZEP/hawor_${NN}_result.npz --target $B/targets/$ZEP/${NN}_trajectory.npz --up $E/up.json --side $SIDE --output $E/bundle_$KK | tail -1 | cut -c1-120; done; echo RC_DONE
