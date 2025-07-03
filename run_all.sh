#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

### C3VD

# Define the base path for your sequences
base_path="/path/to/data/C3VD"

# List of sequences
sequences=(
    "cecum_t1_b"
    "cecum_t2_b"
    "cecum_t3_a"
    "sigmoid_t1_a"
    "sigmoid_t2_a"
    "sigmoid_t3_a"
    "trans_t1_b"
    "trans_t2_c"
    "trans_t4_a"
    "trans_t4_b"
)

# Loop through each sequence and run the commands
for seq in "${sequences[@]}"; do
    sequence_path="$base_path/$seq"
    model_path="output_test/$seq"  # Use sequence name as part of the model path
    echo "Processing sequence: $sequence_path with model path: $model_path"


    python train_viewdir.py --model_path "$model_path" --eval --iteration 40000 -s "$sequence_path" --lambda_depth 0.2 --lambda_norm 0.1 --albedo_loss_weight 1.0 \
    --densification_interval 1000 --opacity_reset_interval 3000 --densify_until_iter 10000 --densify_from_iter 1000 --mlp_lr 0.0005 --grid_lr 0.0005 \
    --end_diffuse_loss_iter 20000 --port 6000 --K_normals 100
    python render.py --model_path "$model_path" --eval -s "$sequence_path" --skip_train --iteration 40000
    python render_body_movement.py --model_path "$model_path" --eval -s "$sequence_path" --skip_train --iteration 40000
    python render_light_rotations.py --iteration 40000 --model_path "$model_path" --eval -s "$sequence_path" --spotlight_attenuation_power 10 --max_rot 12
    
done

# compute mean metric values based on logs
python metrics.py "output_test"


### RotateColon
sequence_path="path/to/data/RotateColon"
model_path="output_RotateColon"  # Use sequence name as part of the model path
    

python train_viewdir.py --model_path "$model_path" --eval --iteration 30000 -s "$sequence_path" --lambda_depth 0.2 --lambda_norm 0.1 --albedo_loss_weight 1.0 \
--densification_interval 1000 --opacity_reset_interval 3000 --densify_until_iter 10000 --densify_from_iter 1000 --mlp_lr 0.0005 --grid_lr 0.0005 \
--end_diffuse_loss_iter 20000 --port 6000 --K_normals 30 --diffuse_loss_weight 0.01
