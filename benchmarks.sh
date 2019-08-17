#!/bin/bash
work_group_size=8
kernel_dim=9
img_path="./input/img1.jpg"
# change kernel_dimensions


for k_dim in 5 9 13 17 21
do
    echo "Kernel Naive >> - Kernel-Dim-"$k_dim
    PYOPENCL_COMPILER_OUTPUT=1 python main.py $img_path kernel_naive $work_group_size $k_dim >> ./output_k_dim/output_naive_kernel_dim_$k_dim.csv

    echo "Kernel Uchar4 >> Kernel-Dim-"$k_dim
    PYOPENCL_COMPILER_OUTPUT=1 python main.py $img_path kernel_uchar4 $work_group_size $k_dim>> ./output_k_dim/output_uchar4_kernel_dim_$k_dim.csv

    echo "Kernel Local Memory >> Kernel-Dim-"$k_dim
    PYOPENCL_COMPILER_OUTPUT=1 python main_local.py $img_path kernel_uchar4_local_mem $work_group_size $k_dim >> ./output_k_dim/output_local_mem_kernel_dim_$k_dim.csv

done


for wg_size in 4 8 16
do
    echo "Kernel Naive >> work group - "$wg_size
    PYOPENCL_COMPILER_OUTPUT=1 python main.py $img_path kernel_naive $wg_size $kernel_dim >> ./output_wgsize/output_naive_wgs_$wg_size.csv

    echo "Kernel Uchar4 >> work group - "$wg_size
    PYOPENCL_COMPILER_OUTPUT=1 python main.py $img_path kernel_uchar4 $wg_size $kernel_dim >> ./output_wgsize/output_uchar4_wgs_$wg_size.csv

    echo "Kernel Local Memory >> work group - "$wg_size
    PYOPENCL_COMPILER_OUTPUT=1 python main_local.py $img_path kernel_uchar4_local_mem $wg_size $kernel_dim >> ./output_wgsize/output_local_mem_wgs_$wg_size.csv

done

for img in 0 1 2
do

	echo "Kernel Naive >> Image - "$img
    PYOPENCL_COMPILER_OUTPUT=1 python main.py ./input/img$img.jpg kernel_naive $work_group_size $kernel_dim >> ./output_img/output_naive_img_$img.csv

    echo "Kernel Uchar4 >> Image - "$img
    PYOPENCL_COMPILER_OUTPUT=1 python main.py ./input/img$img.jpg kernel_uchar4 $work_group_size $kernel_dim  >> ./output_img/output_uchar4_img_$img.csv

    echo "Kernel Local Memory >> Image - "$img
    PYOPENCL_COMPILER_OUTPUT=1 python main_local.py ./input/img$img.jpg kernel_uchar4_local_mem $work_group_size $kernel_dim >> ./output_img/output_local_mem_img_$img.csv
done 

