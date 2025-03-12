#include <cstdlib>  // For malloc, free
#include <iostream> // For std::cout, std::cerr, std::endl
#include <cassert>  // For assert

#include "tomo_common.h"


void
art(const float* data, int dy, int dt, int dx, const float* center, const float* theta,
    float* recon, int ngridx, int ngridy, int num_iter)
{
    if(dy == 0 || dt == 0 || dx == 0)
        return;

    float* gridx   = (float*) malloc((ngridx + 1) * sizeof(float));
    float* gridy   = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordx  = (float*) malloc((ngridy + 1) * sizeof(float));
    float* coordy  = (float*) malloc((ngridx + 1) * sizeof(float));
    float* ax      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* ay      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* bx      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* by      = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coorx   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* coory   = (float*) malloc((ngridx + ngridy) * sizeof(float));
    float* dist    = (float*) malloc((ngridx + ngridy) * sizeof(float));
    int*   indi    = (int*) malloc((ngridx + ngridy) * sizeof(int));
    float* simdata = (float*) malloc((dy * dt * dx) * sizeof(float));

    assert(coordx != NULL && coordy != NULL && ax != NULL && ay != NULL && by != NULL &&
           bx != NULL && coorx != NULL && coory != NULL && dist != NULL && indi != NULL &&
           simdata != NULL);

    int   s, p, d, i, n;
    int   quadrant;
    float theta_p, sin_p, cos_p;
    float mov, xi, yi;
    int   asize, bsize, csize;
    float upd;
    int   ind_data, ind_recon;

    for(i = 0; i < num_iter; i++)
    {

        // std::cout << "Iteration: " << i << std::endl;

        // initialize simdata to zero
        memset(simdata, 0, dy * dt * dx * sizeof(float));

        preprocessing(ngridx, ngridy, dx, center[0], &mov, gridx,
                      gridy);  // Outputs: mov, gridx, gridy

        // For each projection angle
        for(p = 0; p < dt; p++)
        {

            // Calculate the sin and cos values
            // of the projection angle and find
            // at which quadrant on the cartesian grid.
            theta_p  = fmodf(theta[p], 2.0f * (float) M_PI);
            quadrant = calc_quadrant(theta_p);
            sin_p    = sinf(theta_p);
            cos_p    = cosf(theta_p);

            // For each detector pixel
            for(d = 0; d < dx; d++)
            {
                // Calculate coordinates
                xi = -ngridx - ngridy;
                yi = 0.5f * (1 - dx) + d + mov;
                calc_coords(ngridx, ngridy, xi, yi, sin_p, cos_p, gridx, gridy, coordx,
                            coordy);

                // Merge the (coordx, gridy) and (gridx, coordy)
                trim_coords(ngridx, ngridy, coordx, coordy, gridx, gridy, &asize, ax, ay,
                            &bsize, bx, by);

                // Sort the array of intersection points (ax, ay) and
                // (bx, by). The new sorted intersection points are
                // stored in (coorx, coory). Total number of points
                // are csize.
                sort_intersections(quadrant, asize, ax, ay, bsize, bx, by, &csize, coorx,
                                   coory);

                // Calculate the distances (dist) between the
                // intersection points (coorx, coory). Find the
                // indices of the pixels on the reconstruction grid.
                calc_dist(ngridx, ngridy, csize, coorx, coory, indi, dist);

                // Calculate dist*dist
                float sum_dist2 = 0.0f;
                for(n = 0; n < csize - 1; n++)
                {
                    sum_dist2 += dist[n] * dist[n];
                }

                if(sum_dist2 != 0.0f)
                {
                    // For each slice
                    for(s = 0; s < dy; s++)
                    {
                        // Calculate simdata
                        calc_simdata(s, p, d, ngridx, ngridy, dt, dx, csize, indi, dist,
                                     recon,
                                     simdata);  // Output: simdata

                        // Update
                        ind_data  = d + p * dx + s * dt * dx;
                        ind_recon = s * ngridx * ngridy;
                        upd       = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                        for(n = 0; n < csize - 1; n++)
                        {
                            recon[indi[n] + ind_recon] += upd * dist[n];
                        }
                    }
                }
            }
        }
    }
    free(gridx);
    free(gridy);
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    free(coorx);
    free(coory);
    free(dist);
    free(indi);
    free(simdata);
}
