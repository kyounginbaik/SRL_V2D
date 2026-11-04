# Reconstruction recovery evidence

Public reference data is used only for scoring. Image/depth residuals are fitted observations, not hidden-GT certification.

|split|ep|frames|camera valid|FP objects|point-map observations|
|---|---:|---:|---:|---:|---|
|public|41|299|1.000|1/1|white_pot 0.75|
|public|44|249|1.000|1/1|white_pot 0.92|
|public|39|319|1.000|1/1|mini_dust_pan 0.37|
|public|11|137|1.000|2/2|blue_cup 0.80, beige_cup 0.94|
|public|18|353|1.000|2/2|mini_sweeper 0.86, mini_dust_pan 0.82|
|public|23|325|1.000|2/2|wooden_piece_1 0.20, wooden_piece_2 0.89|
|public|31|290|1.000|2/2|mini_sweeper 0.74, plastic_dish_rack 1.00|
|public|12|181|1.000|2/2|blue_cup 0.24, plastic_dish_rack 1.00|
|public|21|186|1.000|2/2|blue_cup 0.91, water_pitcher 0.58|
|public|24|275|0.985|2/2|wooden_piece_1 0.25, wooden_piece_2 0.77|
|evaluation|1|335|1.000|2/2||
|evaluation|3|313|1.000|2/2|white_pot 0.94, wooden_spoon 0.20|
|evaluation|5|288|1.000|2/2|white_pot 0.66, wooden_spoon 0.42|
|evaluation|6|243|1.000|2/2|white_pot 0.75, wooden_spoon 0.47|
|evaluation|8|219|1.000|2/2|beige_cup 0.50, white_pot 0.98|
|evaluation|10|161|1.000|2/2|blue_cup 0.90, beige_cup 0.36|
|evaluation|14|158|1.000|2/2|blue_cup 0.18, plastic_dish_rack 1.00|
|evaluation|15|166|1.000|2/2|plastic_dish_rack 1.00, wooden_spoon 0.33|
|evaluation|16|173|1.000|2/2|plastic_dish_rack 1.00, wooden_spoon 0.32|
|evaluation|19|235|0.996|2/2|mini_sweeper 0.41, mini_dust_pan 0.64|
|evaluation|20|202|1.000|2/2|blue_cup 0.98, water_pitcher 0.60|
|evaluation|25|356|1.000|2/2||
|evaluation|28|220|1.000|2/2|plastic_dish_rack 1.00, white_pot 0.33|
|evaluation|30|263|1.000|2/2|mini_sweeper 0.72, plastic_dish_rack 1.00|
|evaluation|32|259|1.000|2/2|blue_cup 0.85, wooden_spoon 0.50|
|evaluation|33|348|0.991|2/2|blue_cup 0.61, wooden_spoon 0.54|
|evaluation|35|421|1.000|1/1|wooden_spoon 0.44|
|evaluation|36|407|1.000|1/1|wooden_spoon 0.23|
|evaluation|37|151|1.000|1/1|wooden_spoon 0.99|
|evaluation|38|263|1.000|1/1|mini_dust_pan 0.65|

## Public validation

|ep|variant|errors (cm / deg)|MP-SR|
|---:|---|---|---:|
|41|pointmap_cotracker|white_pot 1.75 / 15.9|1.0|
|41|raw|white_pot 0.78 / 3.6|1.0|
|44|pointmap_cotracker|white_pot 0.61 / 3.0|1.0|
|44|raw|white_pot 0.56 / 1.8|1.0|
|39|pointmap_cotracker|mini_dust_pan 3.54 / 24.4|0.0|
|39|raw|mini_dust_pan 0.96 / 3.8|1.0|
|11|cup_continuity|blue_cup 3.80 / 6.4; beige_cup 0.86 / 2.8|0.0|
|11|dense|blue_cup 1.30 / 2.6; beige_cup 3.41 / 3.0|0.0|
|11|multimap|blue_cup 0.31 / 1.3; beige_cup 0.98 / 7.0|1.0|
|11|pointmap_cotracker|blue_cup 0.41 / 1.8; beige_cup 1.04 / 9.9|1.0|
|11|pointmap_lk|blue_cup 0.52 / 3.2; beige_cup 12.48 / 10.1|0.0|
|11|raw|blue_cup 3.43 / 92.9; beige_cup 0.86 / 2.8|0.0|
|18|dense_markers|mini_sweeper 1.28 / 10.1; mini_dust_pan 11.76 / 27.3|0.0|
|18|multimap|mini_sweeper 6.29 / 20.4; mini_dust_pan 46.04 / 176.3|0.0|
|18|multimap_rgbd_markers|mini_sweeper 6.30 / 20.5; mini_dust_pan 46.01 / 176.9|0.0|
|18|multimap_rgbd_markers_clean|mini_sweeper 0.98 / 8.7; mini_dust_pan 2.18 / 7.6|1.0|
|18|pointmap_cotracker|mini_sweeper 6.17 / 21.1; mini_dust_pan 45.87 / 176.2|0.0|
|18|pointmap_cotracker_rgbd_markers|mini_sweeper 6.18 / 21.2; mini_dust_pan 45.84 / 176.7|0.0|
|18|pointmap_cotracker_rgbd_markers_clean|mini_sweeper 1.05 / 9.1; mini_dust_pan 2.22 / 8.6|1.0|
|18|pointmap_cotracker_surface|mini_sweeper 6.17 / 21.1; mini_dust_pan 45.86 / 177.0|0.0|
|18|raw|mini_sweeper 6.72 / 141.1; mini_dust_pan 46.17 / 177.0|0.0|
|18|raw_rgbd_markers|mini_sweeper 6.77 / 141.0; mini_dust_pan 46.14 / 177.5|0.0|
|18|raw_rgbd_markers_clean|mini_sweeper 2.01 / 144.5; mini_dust_pan 2.06 / 6.8|0.0|
|18|raw_surface|mini_sweeper 6.72 / 141.0; mini_dust_pan 46.16 / 177.8|0.0|
|23|observed_markers_v3|wooden_piece_1 0.88 / 2.9; wooden_piece_2 1.08 / 2.6|1.0|
|23|pointmap_cotracker|wooden_piece_1 5.68 / 36.6; wooden_piece_2 1.54 / 28.4|0.0|
|23|raw|wooden_piece_1 1.03 / 3.3; wooden_piece_2 1.25 / 179.3|0.0|
|31|multimap|mini_sweeper 1.00 / 8.7; plastic_dish_rack 1.03 / 5.8|1.0|
|31|pointmap_cotracker|mini_sweeper 1.63 / 9.9; plastic_dish_rack 0.99 / 5.6|1.0|
|31|pointmap_cotracker_rgbd_markers|mini_sweeper 1.66 / 9.8; plastic_dish_rack 1.15 / 5.8|1.0|
|31|pointmap_cotracker_rgbd_markers_clean|mini_sweeper 14.84 / 29.8; plastic_dish_rack 22.01 / 179.7|0.0|
|31|raw|mini_sweeper 1.15 / 7.7; plastic_dish_rack 1.39 / 5.7|1.0|
|31|raw_rgbd_markers|mini_sweeper 1.19 / 7.7; plastic_dish_rack 1.53 / 5.8|1.0|
|31|raw_rgbd_markers_clean|mini_sweeper 15.62 / 34.0; plastic_dish_rack 22.29 / 179.6|0.0|
|12|cup_continuity|blue_cup 1.51 / 5.9; plastic_dish_rack 2.16 / 2.4|1.0|
|12|multimap|blue_cup 16.47 / 72.4; plastic_dish_rack 2.21 / 0.9|0.0|
|12|pointmap_cotracker|blue_cup 9.70 / 110.5; plastic_dish_rack 2.17 / 0.8|0.0|
|12|raw|blue_cup 1.51 / 5.9; plastic_dish_rack 2.16 / 2.4|1.0|
|21|cup_continuity|blue_cup 0.85 / 2.8; water_pitcher 1.54 / 4.0|1.0|
|21|multimap|blue_cup 1.07 / 5.2; water_pitcher 2.53 / 7.4|1.0|
|21|pointmap_cotracker|blue_cup 1.00 / 6.1; water_pitcher 2.68 / 12.8|1.0|
|21|raw|blue_cup 0.85 / 2.8; water_pitcher 1.54 / 4.0|1.0|
|24|multimap|wooden_piece_1 20.00 / 110.8; wooden_piece_2 44.49 / 176.0|0.0|
|24|multimap_markers_v3|wooden_piece_1 4.95 / 34.9; wooden_piece_2 2.49 / 24.1|0.0|
|24|observed_markers|wooden_piece_1 1.50 / 3.1; wooden_piece_2 1.68 / 9.0|1.0|
|24|observed_markers_v2|wooden_piece_1 1.47 / 2.5; wooden_piece_2 1.60 / 9.1|1.0|
|24|observed_markers_v3|wooden_piece_1 1.60 / 2.8; wooden_piece_2 1.63 / 9.1|1.0|
|24|pointmap_cotracker|wooden_piece_1 8.43 / 43.7; wooden_piece_2 1.77 / 10.8|0.0|
|24|raw|wooden_piece_1 23.06 / 135.0; wooden_piece_2 45.13 / 174.3|0.0|
