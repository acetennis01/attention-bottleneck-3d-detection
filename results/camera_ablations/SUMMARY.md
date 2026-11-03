# Camera ablation evaluations

Black/shuffled images test sensitivity to out-of-distribution camera corruptions. A drop supports camera dependence, not statistical significance or attribution of the entire gain over a LiDAR baseline. Scores and differences below use percentage/AP points.

| Model | Camera | mAP / 3D AP40 moderate | NDS / BEV AP40 moderate |
|---|---|---:|---:|
| nuscenes_transfusion_swint_b8 | correct | 64.7950 | 69.3897 |
| nuscenes_transfusion_swint_b8 | black | 61.8404 | 67.6933 |
| nuscenes_transfusion_swint_b8 | shuffled | 63.3732 | 68.6699 |
| kitti_v2_b8 | correct | 44.3949 | 55.0548 |
| kitti_v2_b8 | black | 42.7613 | 53.3524 |
| kitti_v2_b8 | shuffled | 42.5492 | 52.3017 |
| kitti_v2_b16 | correct | 43.2628 | 54.2189 |
| kitti_v2_b16 | black | 41.5871 | 52.4950 |
| kitti_v2_b16 | shuffled | 41.0570 | 51.6586 |
| nuscenes_multiscale_b8 | correct | 30.1665 | 46.2146 |
| nuscenes_multiscale_b8 | black | 20.0538 | 38.5619 |
| nuscenes_multiscale_b8 | shuffled | 22.5434 | 42.6095 |
| nuscenes_original_b4 | correct | 27.7797 | 45.2946 |
| nuscenes_original_b4 | black | 24.6637 | 44.0536 |
| nuscenes_original_b4 | shuffled | 25.5891 | 44.2420 |
| kitti_legacy_b1 | correct | 42.7868 | 53.8465 |
| kitti_legacy_b1 | black | 42.0223 | 53.4780 |
| kitti_legacy_b1 | shuffled | 41.7732 | 53.5323 |
| kitti_legacy_b2 | correct | 43.0612 | 54.8251 |
| kitti_legacy_b2 | black | 42.8080 | 54.9611 |
| kitti_legacy_b2 | shuffled | 42.7756 | 54.7275 |
| kitti_legacy_b4 | correct | 42.6099 | 53.9047 |
| kitti_legacy_b4 | black | 41.8772 | 53.6188 |
| kitti_legacy_b4 | shuffled | 41.9688 | 53.3515 |
| kitti_legacy_b8 | correct | 43.2288 | 54.0672 |
| kitti_legacy_b8 | black | 41.2687 | 53.3367 |
| kitti_legacy_b8 | shuffled | 41.7044 | 53.6384 |
| kitti_legacy_b16 | correct | 43.0773 | 55.5481 |
| kitti_legacy_b16 | black | 43.1063 | 55.4231 |
| kitti_legacy_b16 | shuffled | 42.5736 | 55.0755 |
