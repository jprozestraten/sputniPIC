#ifndef INTERPDENSSPECIES_H
#define INTERPDENSSPECIES_H

#include "Alloc.h"
#include "PrecisionTypes.h"
#include "Grid.h"

/** Interpolated densities per species on nodes */
struct interpDensSpecies {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    // index 1: rho
    FPinterp*** rhon; FPinterp *rhon_flat;
    FPinterp*** rhoc; FPinterp *rhoc_flat;
    
    // index 2, 3, 4
    FPinterp*** Jx; FPinterp *Jx_flat;
    FPinterp*** Jy; FPinterp *Jy_flat;
    FPinterp*** Jz; FPinterp *Jz_flat;
    // index 5, 6, 7, 8, 9, 10: pressure tensor (symmetric)
    FPinterp*** pxx; FPinterp *pxx_flat;
    FPinterp*** pxy; FPinterp *pxy_flat;
    FPinterp*** pxz; FPinterp *pxz_flat;
    FPinterp*** pyy; FPinterp *pyy_flat;
    FPinterp*** pyz; FPinterp *pyz_flat;
    FPinterp*** pzz; FPinterp *pzz_flat;
    
};

struct interpDensSpecies_a {

    FPinterp* rhon_flat; FPinterp* rhoc_flat; FPinterp* Jx_flat; FPinterp* Jy_flat; FPinterp* Jz_flat;
    FPinterp* pxx_flat; FPinterp* pxy_flat; FPinterp* pxz_flat; FPinterp* pyy_flat; FPinterp* pyz_flat; 
    FPinterp* pzz_flat;

};

enum copy_way{CPU_TO_GPU, GPU_TO_CPU};

void interp_dens_species_deallocate_gpu(struct interpDensSpecies_a* ids_gpu);

void interp_dens_species_allocate_gpu(struct interpDensSpecies_a* ids_gpu, struct grid* grd);

void ids_copy(struct interpDensSpecies* ids, struct interpDensSpecies_a* ids_gpu, struct grid* grd, copy_way c);


/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is);

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids);

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd);

#endif
