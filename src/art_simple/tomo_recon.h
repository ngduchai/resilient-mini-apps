// Create header file for tomo_recon.cc

#ifndef TOMO_RECON_H
#define TOMO_RECON_H

void art(const float* data, 
         int dy, int dt, int dx, 
         const float* center, 
         const float* theta, 
         float* recon, 
         int ngridx, int ngridy, 
         int num_iter);

void mlem(const float* data, 
         int dy, int dt, int dx, 
         const float* center, 
         const float* theta, 
         float* recon, 
         int ngridx, int ngridy, 
         int num_iter);

void sirt(const float* data, 
         int dy, int dt, int dx, 
         const float* center, 
         const float* theta, 
         float* recon, 
         int ngridx, int ngridy, 
         int num_iter);

void recon_simple(std::string method,
        const float* data,
        int dy, int dt, int dx, 
        const float* center,
        const float* theta,
        float* recon,
        int ngridx, int ngridy,
        int num_iter);
    

#endif // TOMO_RECON_H
