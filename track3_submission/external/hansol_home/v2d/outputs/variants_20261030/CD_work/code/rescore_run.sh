W=/home/hansol/v2d/outputs/variants_20261030/CD_work
for se in public/episode_000021 public_heldout/episode_000022; do
  for t in r8_base g15_camrt; do bash $W/code/score.sh $W/pub/$se/exports/$t; done
done
echo DONE
