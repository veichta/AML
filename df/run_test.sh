python main.py --gpu_id 7 \
            --gpu_mem 11 \
            --estimate ber \
            --verbose 1 \
            --epochs 2 \
            --data_path /mnt/ds3lab-scratch/veichta/Datasets/AWF_dataset/Dynaflow \
            --model_path /mnt/ds3lab-scratch/veichta/Models/AWF/Dynaflow \
            --embedding_path /mnt/ds3lab-scratch/veichta/Embeddings/AWF/Dynaflow \
            --features cw_defended-closed-1_training cw_defended-closed-10_training\
            --use_wandb 0 \
            --log_group Dynaflow \
            --log_dir logs/
            
