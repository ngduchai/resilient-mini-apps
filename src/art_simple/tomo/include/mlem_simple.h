// Create header file for mlem_simple.cc

#ifndef MLEM_SIMPLE_H
#define MLEM_SIMPLE_H

void mlem(const float* data, 
         int dy, int dt, int dx, 
         const float* center, 
         const float* theta, 
         float* recon, 
         int ngridx, int ngridy, 
         int num_iter);

#endif // MLEM_SIMPLE_H
