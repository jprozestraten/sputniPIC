#include "InterpDensSpecies.h"



void interp_dens_species_deallocate_gpu(struct interpDensSpecies_a* ids_gpu)
{
    cudaFree(&(ids_gpu->rhon_flat));
    cudaFree(&(ids_gpu->rhoc_flat));
    cudaFree(&(ids_gpu->Jx_flat));
    cudaFree(&(ids_gpu->Jy_flat));
    cudaFree(&(ids_gpu->Jz_flat));
    cudaFree(&(ids_gpu->pxx_flat));
    cudaFree(&(ids_gpu->pxy_flat));
    cudaFree(&(ids_gpu->pxz_flat));
    cudaFree(&(ids_gpu->pyy_flat));
    cudaFree(&(ids_gpu->pyz_flat));
    cudaFree(&(ids_gpu->pzz_flat));
} 


void interp_dens_species_allocate_gpu(struct interpDensSpecies_a* ids_gpu, struct grid* grd)
{
    FPinterp* d_ids[11]; 
    for (int i = 0; i < 10; ++i)
        cudaMalloc(&d_ids[i], grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp));

    ids_gpu->rhon_flat = d_ids[0];
    ids_gpu->Jx_flat = d_ids[1];
    ids_gpu->Jy_flat = d_ids[2];
    ids_gpu->Jz_flat = d_ids[3];
    ids_gpu->pxx_flat = d_ids[4];
    ids_gpu->pxy_flat = d_ids[5];
    ids_gpu->pxz_flat = d_ids[6];
    ids_gpu->pyy_flat = d_ids[7];
    ids_gpu->pyz_flat = d_ids[8];
    ids_gpu->pzz_flat = d_ids[9];

    cudaMalloc(&d_ids[10], grd->nxc*grd->nyc*grd->nzc*sizeof(FPinterp));

    ids_gpu->rhoc_flat = d_ids[10];

}

void ids_copy(struct interpDensSpecies* ids, struct interpDensSpecies_a* ids_gpu, struct grid* grd, copy_way c)
{
    if (c == CPU_TO_GPU)
    {
        cudaMemcpy(ids_gpu->rhon_flat, ids->rhon_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->rhoc_flat, ids->rhoc_flat, grd->nxc*grd->nyc*grd->nzc*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->Jx_flat, ids->Jx_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->Jy_flat, ids->Jy_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->Jz_flat, ids->Jz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->pxx_flat, ids->pxx_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->pxy_flat, ids->pxy_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->pxz_flat, ids->pxz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->pyy_flat, ids->pyy_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->pyz_flat, ids->pyz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_gpu->pzz_flat, ids->pzz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMemcpy(ids->rhon_flat, ids_gpu->rhon_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->rhoc_flat, ids_gpu->rhoc_flat, grd->nxc*grd->nyc*grd->nzc*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->Jx_flat, ids_gpu->Jx_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->Jy_flat, ids_gpu->Jy_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->Jz_flat, ids_gpu->Jz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->pxx_flat, ids_gpu->pxx_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->pxy_flat, ids_gpu->pxy_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->pxz_flat, ids_gpu->pxz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->pyy_flat, ids_gpu->pyy_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->pyz_flat, ids_gpu->pyz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids->pzz_flat, ids_gpu->pzz_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPinterp), cudaMemcpyDeviceToHost);
    }

}


/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is)
{
    // set species ID
    ids->species_ID = is;
    
    // allocate 3D arrays
    // rho: 1
    ids->rhon = newArr3<FPinterp>(&ids->rhon_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    ids->rhoc = newArr3<FPinterp>(&ids->rhoc_flat, grd->nxc, grd->nyc, grd->nzc); // center
    // Jx: 2
    ids->Jx   = newArr3<FPinterp>(&ids->Jx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jy: 3
    ids->Jy   = newArr3<FPinterp>(&ids->Jy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Jz: 4
    ids->Jz   = newArr3<FPinterp>(&ids->Jz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxx: 5
    ids->pxx  = newArr3<FPinterp>(&ids->pxx_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxy: 6
    ids->pxy  = newArr3<FPinterp>(&ids->pxy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pxz: 7
    ids->pxz  = newArr3<FPinterp>(&ids->pxz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyy: 8
    ids->pyy  = newArr3<FPinterp>(&ids->pyy_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pyz: 9
    ids->pyz  = newArr3<FPinterp>(&ids->pyz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    // Pzz: 10
    ids->pzz  = newArr3<FPinterp>(&ids->pzz_flat, grd->nxn, grd->nyn, grd->nzn); // nodes
    
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids)
{
    
    // deallocate 3D arrays
    delArr3(ids->rhon, grd->nxn, grd->nyn);
    delArr3(ids->rhoc, grd->nxc, grd->nyc);
    // deallocate 3D arrays: J - current
    delArr3(ids->Jx, grd->nxn, grd->nyn);
    delArr3(ids->Jy, grd->nxn, grd->nyn);
    delArr3(ids->Jz, grd->nxn, grd->nyn);
    // deallocate 3D arrays: pressure
    delArr3(ids->pxx, grd->nxn, grd->nyn);
    delArr3(ids->pxy, grd->nxn, grd->nyn);
    delArr3(ids->pxz, grd->nxn, grd->nyn);
    delArr3(ids->pyy, grd->nxn, grd->nyn);
    delArr3(ids->pyz, grd->nxn, grd->nyn);
    delArr3(ids->pzz, grd->nxn, grd->nyn);
    
    
}

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd){
    for (register int i = 1; i < grd->nxc - 1; i++)
        for (register int j = 1; j < grd->nyc - 1; j++)
            for (register int k = 1; k < grd->nzc - 1; k++){
                ids->rhoc[i][j][k] = .125 * (ids->rhon[i][j][k] + ids->rhon[i + 1][j][k] + ids->rhon[i][j + 1][k] + ids->rhon[i][j][k + 1] +
                                       ids->rhon[i + 1][j + 1][k]+ ids->rhon[i + 1][j][k + 1] + ids->rhon[i][j + 1][k + 1] + ids->rhon[i + 1][j + 1][k + 1]);
            }
}
