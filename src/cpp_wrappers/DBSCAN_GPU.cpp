//
// Created by jakobrj on 5/31/22.
//

#import <stdio.h>

#include "../algorithms/GPU_DBSCAN.cuh"
#include "DBSCAN_GPU.h"

void GPU_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts) {
    GPU_DBSCAN(h_C, h_data, n, d, eps, minPts);
}

void G_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts) {
    G_DBSCAN(h_C, h_data, n, d, eps, minPts);
}