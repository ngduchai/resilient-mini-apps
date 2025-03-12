// Create header file for tomo_common.cc

#ifndef TOMO_COMMON_H
#define TOMO_COMMON_H


void preprocessing(int ry, int rz, int num_pixels, float center,
                float* mov, float* gridx, float* gridy);

int calc_quadrant(float theta_p);

void calc_coords(int ry, int rz, float xi, float yi, float sin_p, float cos_p,
            const float* gridx, const float* gridy, float* coordx, float* coordy);

void trim_coords(int ry, int rz, const float* coordx, const float* coordy,
            const float* gridx, const float* gridy, int* asize, float* ax, float* ay,
            int* bsize, float* bx, float* by);

void sort_intersections(int ind_condition, int asize, const float* ax, const float* ay,
                   int bsize, const float* bx, const float* by, int* csize, float* coorx,
                   float* coory);

void calc_dist(int ry, int rz, int csize, const float* coorx, const float* coory,
                    int* indi, float* dist);

void calc_dist2(int ry, int rz, int csize, const float* coorx, const float* coory,
                    int* indx, int* indy, float* dist);

void calc_simdata(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
             const int* indi, const float* dist, const float* model, float* simdata);

void calc_simdata2(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
              const int* indx, const int* indy, const float* dist, float vx, float vy,
              const float* modelx, const float* modely, float* simdata);

void calc_simdata3(int s, int p, int d, int ry, int rz, int dt, int dx, int csize,
              const int* indx, const int* indy, const float* dist, float vx, float vy,
              const float* modelx, const float* modely, const float* modelz, int axis,
              float* simdata);

#endif // TOMO_COMMON_H
