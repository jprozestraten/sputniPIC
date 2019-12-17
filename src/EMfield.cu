#include "EMfield.h"


void init_field_gpu(struct EMfield* field, struct EMfield* field_gpu, struct grid* grd)
{
    /* Electric field defined on nodes: last index is component */
    cudaMemcpy(field_gpu->Ex_flat, field->Ex_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    field_gpu->Ex = NULL;
    cudaMemcpy(field_gpu->Ey_flat, field->Ey_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    field_gpu->Ey = NULL;
    cudaMemcpy(field_gpu->Ez_flat, field->Ez_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    field_gpu->Ez = NULL;
    /* Magnetic field defined on nodes: last index is component */
    field_gpu->Bxn = NULL;
    cudaMemcpy(field_gpu->Bxn_flat, field->Bxn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    field_gpu->Byn = NULL;
    cudaMemcpy(field_gpu->Byn_flat, field->Byn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    field_gpu->Bzn = NULL;
    cudaMemcpy(field_gpu->Bzn_flat, field->Bzn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);

}



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
