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

echo "done"

exit;


# download.sh

python run_alphafold.py \
    --json_path=/home/zhongz2/alphafold_input.json \
    --model_dir=/data/zhongz2/alphafold3_data/models \
    --output_dir=/data/zhongz2/alphafold3_data/af_output
















