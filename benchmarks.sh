echo "Kernel Naive >>"
PYOPENCL_COMPILER_OUTPUT=1 python main.py /Users/ngugi/Desktop/sonic.jpg kernel_naive

echo "Kernel Uchar4 >>"
PYOPENCL_COMPILER_OUTPUT=1 python main.py /Users/ngugi/Desktop/sonic.jpg kernel_uchar4

echo "Kernel Local Memory >>"
PYOPENCL_COMPILER_OUTPUT=1 python main_local.py /Users/ngugi/Desktop/sonic.jpg kernel_uchar4_local_mem 16
