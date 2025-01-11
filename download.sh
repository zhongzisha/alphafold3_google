#!/bin/bash

#SBATCH --mail-type=FAIL

current_dir=`pwd`

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    if [ -d /tmp/zhongz2/data ]; then rm -rf /tmp/zhongz2/data; fi
    CACHE_ROOT=/tmp/zhongz2/$SLURM_JOB_ID
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CACHE_ROOT=/lscratch/$SLURM_JOB_ID
fi
# export OMP_NUM_THREADS=8

cd ~/alphafold3
python3 fetch_databases.py --download_destination /data/zhongz2/alphafold3_data

exit;

sbatch --ntasks=1 --tasks-per-node=1 --partition=norm --cpus-per-task=2 --time=108:00:00 --mem=8G download.sh


