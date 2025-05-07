- To run sweep using Wandb

  ```bash
  # generate sweep ID (only once)
  wandb sweep --project cube_push_anymal_d_osmo scripts/rsl_rl/sweep/cube_push_anymal_d.yaml

  # Above should output the sweep ID
  # Example sweep ID: mayankm96/cube_push_anymal_d_osmo/vdugnnim
  wandb agent <sweep_id>
  ```

- To download a run from Wandb

  ```bash
  python scripts/rsl_rl/play.py --task Object-Push-Anymal-D-v0 \
    --num_envs 32 --wandb_download \
    --wandb_project cube_push_anymal_d_osmo \
    --wandb_run_id z1lb9a3m
  ```
