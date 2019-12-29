#include "EMfield.h"


/*void field_allocate_gpu(struct EMfield* field_gpu, struct grid* grd)
{
    cudaMalloc(&(field_gpu->Ex_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc(&(field_gpu->Ey_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc(&(field_gpu->Ez_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc(&(field_gpu->Bxn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc(&(field_gpu->Byn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc(&(field_gpu->Bzn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
}


void field_deallocate_gpu(struct EMfield* field_gpu)
{
    cudaFree(field_gpu->Ex_flat);
    cudaFree(field_gpu->Ey_flat);
    cudaFree(field_gpu->Ez_flat);
    cudaFree(field_gpu->Bxn_flat);
    cudaFree(field_gpu->Byn_flat);
    cudaFree(field_gpu->Bzn_flat);
} */


/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field)
{
    // E on nodes
    field->Ex  = newArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey  = newArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez  = newArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    // B on nodes
    field->Bxn = newArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // E deallocate 3D arrays
    delArr3(field->Ex, grd->nxn, grd->nyn);
    delArr3(field->Ey, grd->nxn, grd->nyn);
    delArr3(field->Ez, grd->nxn, grd->nyn);

    // B deallocate 3D arrays
    delArr3(field->Bxn, grd->nxn, grd->nyn);
    delArr3(field->Byn, grd->nxn, grd->nyn);
    delArr3(field->Bzn, grd->nxn, grd->nyn);
}
