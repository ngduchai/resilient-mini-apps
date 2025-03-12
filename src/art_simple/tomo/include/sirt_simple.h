// Create header file for sirt_simple.cc

#ifndef SIRT_SIMPLE_H
#define SIRT_SIMPLE_H

void sirt(const float* data, 
         int dy, int dt, int dx, 
         const float* center, 
         const float* theta, 
         float* recon, 
         int ngridx, int ngridy, 
         int num_iter);

#endif // SIRT_SIMPLE_H
