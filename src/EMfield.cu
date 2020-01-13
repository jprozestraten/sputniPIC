#include "EMfield.h"


void field_allocate_gpu(struct EMfield_a* field_gpu, struct grid* grd)
{
    FPfield* d_field[6];

    for (int i = 0; i < 6; i++)
        cudaMalloc(&d_field[i], grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));

    field_gpu->Ex_flat = d_field[0];
    field_gpu->Ey_flat = d_field[1];
    field_gpu->Ez_flat = d_field[2];
    field_gpu->Bxn_flat = d_field[3];
    field_gpu->Byn_flat = d_field[4];
    field_gpu->Bzn_flat = d_field[5];
}


void field_deallocate_gpu(struct EMfield_a* field_gpu)
{
    cudaFree(&(field_gpu->Ex_flat));
    cudaFree(&(field_gpu->Ey_flat));
    cudaFree(&(field_gpu->Ez_flat));
    cudaFree(&(field_gpu->Bxn_flat));
    cudaFree(&(field_gpu->Byn_flat));
    cudaFree(&(field_gpu->Bzn_flat));
} 

void field_copy(struct EMfield* field, struct EMfield_a field_gpu, struct grid* grd)
{
    cudaMemcpy((field_gpu.Ex_flat), (field->Ex_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((field_gpu.Ey_flat), (field->Ey_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((field_gpu.Ez_flat), (field->Ez_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((field_gpu.Bxn_flat), (field->Bxn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((field_gpu.Byn_flat), (field->Byn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((field_gpu.Bzn_flat), (field->Bzn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);

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
