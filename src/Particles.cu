#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>


void particle_deallocate_gpu(struct particles_a* part_gpu)
{
    cudaFree(&(part_gpu->x));
    cudaFree(&(part_gpu->y));
    cudaFree(&(part_gpu->z));
    cudaFree(&(part_gpu->u));
    cudaFree(&(part_gpu->v));
    cudaFree(&(part_gpu->w));
} 


/** allocate particle arrays */

void particle_allocate_gpu(struct particles* part, struct particles_a* part_gpu)
{
    cudaMalloc(&(part_gpu->x), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->y), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->z), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->v), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->w), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->u), part->npmax*sizeof(FPpart));
}

void particle_copy(struct particles* part, struct particles_a* part_gpu, copy_way c)
{
    if (c == CPU_TO_GPU)
    {
        cudaMemcpy(&(part_gpu->x), &(part->x), part->npmax, cudaMemcpyHostToDevice);
        cudaMemcpy(&(part_gpu->y), &(part->x), part->npmax, cudaMemcpyHostToDevice);
        cudaMemcpy(&(part_gpu->z), &(part->x), part->npmax, cudaMemcpyHostToDevice);
        cudaMemcpy(&(part_gpu->w), &(part->x), part->npmax, cudaMemcpyHostToDevice);
        cudaMemcpy(&(part_gpu->u), &(part->x), part->npmax, cudaMemcpyHostToDevice);
        cudaMemcpy(&(part_gpu->v), &(part->x), part->npmax, cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMemcpy(&(part->x), &(part_gpu->x),part->npmax, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(part->y), &(part_gpu->y),part->npmax, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(part->z), &(part_gpu->z),part->npmax, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(part->w), &(part_gpu->w),part->npmax, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(part->u), &(part_gpu->u),part->npmax, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(part->v), &(part_gpu->v),part->npmax, cudaMemcpyDeviceToHost);
    }

}

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}






/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}



void gpu_mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,
                            struct particles_a* part_gpu, struct EMfield_a* field_gpu, struct grid_a* grid_gpu)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    for (int is=0; is < param->ns; is++)
    {
        particle_copy(&part[is], &(part_gpu[is]), CPU_TO_GPU);
        move_particle<<<((part[is]).nop + TPB - 1)/TPB, TPB>>>(part, field, grd, param, part_gpu, field_gpu, grid_gpu);
        particle_copy(&part[is], &(part_gpu[is]), GPU_TO_CPU);
    }
    cudaDeviceSynchronize();    
                                          
} // end of the mover




/** particle mover kernel */
__global__ void move_particle(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,
                                        struct particles_a* part_gpu, struct EMfield_a* field_gpu, struct grid_a* grid_gpu)
{
    // print species and subcycling
    //std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        xptilde = part_gpu->x[i];
        yptilde = part_gpu->y[i];
        zptilde = part_gpu->z[i];
        // calculate the average velocity iteratively
        for(int innter=0; innter < part->NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((part_gpu->x[i] - grd->xStart)*grd->invdx);
            iy = 2 +  int((part_gpu->y[i] - grd->yStart)*grd->invdy);
            iz = 2 +  int((part_gpu->z[i] - grd->zStart)*grd->invdz);
            
            // calculate weights
            xi[0]   = part_gpu->x[i] - grd->XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)];
            eta[0]  = part_gpu->y[i] - grd->YN_flat[get_idx(ix, iy-1, iz, grd->nyn, grd->nzn)];
            zeta[0] = part_gpu->z[i] - grd->ZN_flat[get_idx(ix, iy, iz-1, grd->nyn, grd->nzn)];

            xi[1]   = grd->XN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)] - part_gpu->x[i];
            eta[1]  = grd->YN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)] - part_gpu->y[i];
            zeta[1] = grd->ZN_flat[get_idx(ix-1, iy, iz, grd->nyn, grd->nzn)] - part_gpu->z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                    {
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                    }
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        Exl += weight[ii][jj][kk]*field_gpu->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Eyl += weight[ii][jj][kk]*field_gpu->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Ezl += weight[ii][jj][kk]*field_gpu->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];

                        Bxl += weight[ii][jj][kk]*field_gpu->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Byl += weight[ii][jj][kk]*field_gpu->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Bzl += weight[ii][jj][kk]*field_gpu->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    }
                
            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= part_gpu->u[i] + qomdt2*Exl;
            vt= part_gpu->v[i] + qomdt2*Eyl;
            wt= part_gpu->w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            part_gpu->x[i] = xptilde + uptilde*dto2;
            part_gpu->y[i] = yptilde + vptilde*dto2;
            part_gpu->z[i] = zptilde + wptilde*dto2;
                
                
        } // end of iteration
        // update the final position and velocity
        part_gpu->u[i]= 2.0*uptilde - part_gpu->u[i];
        part_gpu->v[i]= 2.0*vptilde - part_gpu->v[i];
        part_gpu->w[i]= 2.0*wptilde - part_gpu->w[i];
        part_gpu->x[i] = xptilde + uptilde*dt_sub_cycling;
        part_gpu->y[i] = yptilde + vptilde*dt_sub_cycling;
        part_gpu->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
        //////////
        //////////
        ////////// BC
                                        
        // X-DIRECTION: BC particles
        if (part_gpu->x[i] > grd->Lx){
            if (param->PERIODICX==true){ // PERIODIC
                part_gpu->x[i] = part_gpu->x[i] - grd->Lx;
            } else { // REFLECTING BC
                part_gpu->u[i] = -part_gpu->u[i];
                part_gpu->x[i] = 2*grd->Lx - part_gpu->x[i];
            }
        }
                                                                        
        if (part_gpu->x[i] < 0){
            if (param->PERIODICX==true){ // PERIODIC
               part_gpu->x[i] = part_gpu->x[i] + grd->Lx;
            } else { // REFLECTING BC
                part_gpu->u[i] = -part_gpu->u[i];
                part_gpu->x[i] = -part_gpu->x[i];
            }
        }
                
            
        // Y-DIRECTION: BC particles
        if (part_gpu->y[i] > grd->Ly){
            if (param->PERIODICY==true){ // PERIODIC
                part_gpu->y[i] = part_gpu->y[i] - grd->Ly;
            } else { // REFLECTING BC
                part_gpu->v[i] = -part_gpu->v[i];
                part_gpu->y[i] = 2*grd->Ly - part_gpu->y[i];
            }
        }
                                                                        
        if (part_gpu->y[i] < 0){
            if (param->PERIODICY==true){ // PERIODIC
                part_gpu->y[i] = part_gpu->y[i] + grd->Ly;
            } else { // REFLECTING BC
                part_gpu->v[i] = -part_gpu->v[i];
                part_gpu->y[i] = -part_gpu->y[i];
            }
        }
                                                                        
        // Z-DIRECTION: BC particles
        if (part_gpu->z[i] > grd->Lz){
            if (param->PERIODICZ==true){ // PERIODIC
                part_gpu->z[i] = part_gpu->z[i] - grd->Lz;
            } else { // REFLECTING BC
                part_gpu->w[i] = -part_gpu->w[i];
                part_gpu->z[i] = 2*grd->Lz - part_gpu->z[i];
            }
        }
                                                                        
        if (part_gpu->z[i] < 0){
            if (param->PERIODICZ==true){ // PERIODIC
                part_gpu->z[i] = part_gpu->z[i] + grd->Lz;
            } else { // REFLECTING BC
                part_gpu->w[i] = -part_gpu->w[i];
                part_gpu->z[i] = -part_gpu->z[i];
            }
        }
                                                                        
            
            
          // end of subcycling
    } // end of one particle
                                                                        
} // end of the mover







/** Interpolation particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
