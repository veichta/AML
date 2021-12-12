python main.py --gpu_id 5 \
            --gpu_mem 11 \
            --estimate all \
            --verbose 0 \
            --epochs 40 \
            --data_path /mnt/ds3lab-scratch/veichta/Datasets/AWF_dataset/Glue \
            --model_path /mnt/ds3lab-scratch/veichta/Models/AWF/Glue \
            --embedding_path /mnt/ds3lab-scratch/veichta/Embeddings/AWF/Glue \
            --features glue_m1 \
	    --use_wandb 0 \
            --log_dir ./logs/Glue/
            
