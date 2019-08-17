#!/bin/bash
work_group_size=8
kernel_dim=9
img_path="./input/img1.jpg"
# change kernel_dimensions




for k_dim in 5 9 13 17 21 25
do
    echo "Kernel Naive >> - Kernel-Dim-CPU-"$k_dim
    python main.py $img_path kernel_naive $work_group_size $k_dim 0 >> ./output_cpu_k_dim/output_cpu_naive_kernel_dim_$k_dim.csv
    echo "Kernel Naive >> - Kernel-Dim-GPU-"$k_dim
    python main.py $img_path kernel_naive $work_group_size $k_dim 1 >> ./output_gpu_k_dim/output_gpu_naive_kernel_dim_$k_dim.csv


    echo "Kernel Uchar4 >> Kernel-Dim-CPU-"$k_dim
    python main.py $img_path kernel_uchar4 $work_group_size $k_dim 0 >> ./output_cpu_k_dim/output_cpu_uchar4_kernel_dim_$k_dim.csv
    echo "Kernel Uchar4 >> Kernel-Dim-GPU-"$k_dim
    python main.py $img_path kernel_uchar4 $work_group_size $k_dim 1 >> ./output_gpu_k_dim/output_gpu_uchar4_kernel_dim_$k_dim.csv

    echo "Kernel Local Memory >> Kernel-Dim-CPU-"$k_dim
    python main_local.py $img_path kernel_uchar4_local_mem $work_group_size $k_dim 0 >> ./output_cpu_k_dim/output_cpu_local_mem_kernel_dim_$k_dim.csv
    echo "Kernel Local Memory >> Kernel-Dim-GPU-"$k_dim
    python main_local.py $img_path kernel_uchar4_local_mem $work_group_size $k_dim 1 >> ./output_gpu_k_dim/output_gpu_local_mem_kernel_dim_$k_dim.csv

done

for img in 1 2 3
do
	echo "Kernel Naive >> Image - CPU "$img
    python main.py ./input/img$img.jpg kernel_naive $work_group_size $kernel_dim 0 >> ./output_cpu_img/output_cpu_naive_img_$img.csv
    echo "Kernel Naive >> Image - GPU"$img
    python main.py ./input/img$img.jpg kernel_naive $work_group_size $kernel_dim 1 >> ./output_gpu_img/output_gpu_naive_img_$img.csv

    echo "Kernel Uchar4 >> Image - CPU "$img
    python main.py ./input/img$img.jpg kernel_uchar4 $work_group_size $kernel_dim  0 >> ./output_cpu_img/output_cpu_uchar4_img_$img.csv
    echo "Kernel Uchar4 >> Image - GPU"$img
    python main.py ./input/img$img.jpg kernel_uchar4 $work_group_size $kernel_dim  1 >> ./output_gpu_img/output_gpu_uchar4_img_$img.csv

    echo "Kernel Local Memory >> Image - CPU "$img
    python main_local.py ./input/img$img.jpg kernel_uchar4_local_mem $work_group_size $kernel_dim 0 >> ./output_cpu_img/output_cpu_local_mem_img_$img.csv
    echo "Kernel Local Memory >> Image - GPU"$img
    python main_local.py ./input/img$img.jpg kernel_uchar4_local_mem $work_group_size $kernel_dim 1 >> ./output_gpu_img/output_gpu_local_mem_img_$img.csv
done 

for wg_size in 4 8 16 32
do
    echo "Kernel Naive >> work group - CPU "$wg_size
    python main.py $img_path kernel_naive $wg_size $kernel_dim 0 >> ./output_cpu_wgsize/output_cpu_naive_wgs_$wg_size.csv
    echo "Kernel Naive >> work group - GPU "$wg_size
    python main.py $img_path kernel_naive $wg_size $kernel_dim 0 >> ./output_gpu_wgsize/output_gpu_naive_wgs_$wg_size.csv

    echo "Kernel Uchar4 >> work group - CPU "$wg_size
    python main.py $img_path kernel_uchar4 $wg_size $kernel_dim 0 >> ./output_cpu_wgsize/output_cpu_uchar4_wgs_$wg_size.csv
     echo "Kernel Uchar4 >> work group - GPU"$wg_size
    python main.py $img_path kernel_uchar4 $wg_size $kernel_dim  1 >> ./output_gpu_wgsize/output_gpu_uchar4_wgs_$wg_size.csv

    echo "Kernel Local Memory >> work group - CPU "$wg_size
    python main_local.py $img_path kernel_uchar4_local_mem $wg_size $kernel_dim 0 >> ./output_cpu_wgsize/output_cpu_local_mem_wgs_$wg_size.csv
    echo "Kernel Local Memory >> work group - GPU "$wg_size
    python main_local.py $img_path kernel_uchar4_local_mem $wg_size $kernel_dim 1 >> ./output_gpu_wgsize/output_gpu_local_mem_wgs_$wg_size.csv

done


echo "Sequential K-dim-"$kernel_dim
python sequential.py $img_path $kernel_dim >> ./baseline.csv

