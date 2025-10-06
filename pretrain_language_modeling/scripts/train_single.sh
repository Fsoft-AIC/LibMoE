echo "Training ..."

export MOE_TYPE="moe_layer_film"
export CUDA_VISIBLE_DEVICES="4"

while true; do
    python3 run.py  /cm/shared/anonymous/moeut_training_code/sweeps/experiments/proposed_method/154M/slimpajama_moe_no_attmoe_154M_film.yaml

    if [ $? -eq 0 ]; then
        echo "Training completed successfully"
        break
    else
        echo "Error detected, restarting training in 10 seconds..."
        sleep 10
    fi
done