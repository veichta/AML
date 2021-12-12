python main.py --gpu_id 6 \
            --gpu_mem 11 \
            --estimate all \
            --verbose 0 \
            --epochs 40 \
            --data_path /mnt/ds3lab-scratch/veichta/Datasets/AWF_dataset \
            --model_path /mnt/ds3lab-scratch/veichta/Models/AWF/Merged \
            --embedding_path /mnt/ds3lab-scratch/veichta/Embeddings/AWF/Merged \
            --features merged_1_training merged_2_training merged_3_training merged_4_training merged_5_training \
            --use_wandb 0 \
            --log_dir ./logs/Merged_timing/merged_v3/
            
