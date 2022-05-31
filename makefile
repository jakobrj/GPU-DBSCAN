install:
	nvcc  --shared -o libDBSCAN_GPU.so \
         		src/cpp_wrappers/DBSCAN_GPU.cpp \
         		src/algorithms/GPU_DBSCAN.cu \
         		src/utils/GPU_utils.cu \
         		-Xcompiler -fPIC
	python3 setup.py install