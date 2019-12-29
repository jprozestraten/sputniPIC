#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>


/*void particle_allocate_gpu(struct particles* part, struct particles* part_gpu)
{
    cudaMalloc(&(part_gpu->x), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->y), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->z), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->u), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->v), part->npmax*sizeof(FPpart));
    cudaMalloc(&(part_gpu->w), part->npmax*sizeof(FPpart));
    //q must have precision of interpolated quantities: typically double. Not used in mover 
    cudaMalloc(&(part_gpu->q), part->npmax*sizeof(FPpart)); 
} 

void particle_deallocate_gpu(struct particles* part_gpu)
{
    cudaFree(part_gpu->x);
    cudaFree(part_gpu->y);
    cudaFree(part_gpu->z);
    cudaFree(part_gpu->u);
    cudaFree(part_gpu->v);
    cudaFree(part_gpu->w);
    cudaFree(part_gpu->q);
} */


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
    /// ALLOCATION partICLE ARRAYS
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


/** particle mover */
__global__ void gpu_mover_PC(struct particles* part_gpu, struct EMfield* field_gpu, struct grid* grd_gpu, struct parameters* param_gpu)
{
    // print species and subcycling
    //std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param_gpu->dt/((double) part_gpu->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part_gpu->qom*dto2/param_gpu->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2*2*2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    int idxx, idyy, idzz, idweight;

    // start subcycling
    for (int i_sub=0; i_sub <  part_gpu->n_sub_cycles; i_sub++){
        // move each particle with new fields
        xptilde = part_gpu->x[i];
        yptilde = part_gpu->y[i];
        zptilde = part_gpu->z[i];
        // calculate the average velocity iteratively
        for(int innter=0; innter < part_gpu->NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((part_gpu->x[i] - grd_gpu->xStart)*grd_gpu->invdx);
            iy = 2 +  int((part_gpu->y[i] - grd_gpu->yStart)*grd_gpu->invdy);
            iz = 2 +  int((part_gpu->z[i] - grd_gpu->zStart)*grd_gpu->invdz);
            
            // calculate weights
            idxx = get_idx(ix-1, iy, iz, grd_gpu->nyn, grd_gpu->nzn);
            idyy = get_idx(ix, iy-1, iz, grd_gpu->nyn, grd_gpu->nzn);
            idzz = get_idx(ix, iy, iz-1, grd_gpu->nyn, grd_gpu->nzn);
            xi[0]   = part_gpu->x[i] - grd_gpu->XN_flat[idxx];
            eta[0]  = part_gpu->y[i] - grd_gpu->YN_flat[idyy];
            zeta[0] = part_gpu->z[i] - grd_gpu->ZN_flat[idzz];
            idxx = get_idx(ix, iy, iz, grd_gpu->nyn, grd_gpu->nzn);
            xi[1]   = grd_gpu->XN_flat[idxx] - part_gpu->x[i];
            eta[1]  = grd_gpu->YN_flat[idxx] - part_gpu->y[i];
            zeta[1] = grd_gpu->ZN_flat[idxx] - part_gpu->z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                    {
                        idweight = get_idx(ii, jj, kk, 2, 2);
                        weight[idweight] = xi[ii] * eta[jj] * zeta[kk] * grd_gpu->invVOL;
                    }
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        idweight = get_idx(ii, jj, kk, 2, 2);
                        idxx = get_idx(ix-ii, iy-jj, iz-kk, grd_gpu->nyn, grd_gpu->nzn);
                        Exl += weight[idweight]*field_gpu->Ex_flat[idxx];
                        Eyl += weight[idweight]*field_gpu->Ey_flat[idxx];
                        Ezl += weight[idweight]*field_gpu->Ez_flat[idxx];

                        Bxl += weight[idweight]*field_gpu->Bxn_flat[idxx];
                        Byl += weight[idweight]*field_gpu->Byn_flat[idxx];
                        Bzl += weight[idweight]*field_gpu->Bzn_flat[idxx];
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
        if (part_gpu->x[i] > grd_gpu->Lx){
            if (param_gpu->PERIODICX==true){ // PERIODIC
                part_gpu->x[i] = part_gpu->x[i] - grd_gpu->Lx;
            } else { // REFLECTING BC
                part_gpu->u[i] = -part_gpu->u[i];
                part_gpu->x[i] = 2*grd_gpu->Lx - part_gpu->x[i];
            }
        }
                                                                        
        if (part_gpu->x[i] < 0){
            if (param_gpu->PERIODICX==true){ // PERIODIC
               part_gpu->x[i] = part_gpu->x[i] + grd_gpu->Lx;
            } else { // REFLECTING BC
                part_gpu->u[i] = -part_gpu->u[i];
                part_gpu->x[i] = -part_gpu->x[i];
            }
        }
                
            
        // Y-DIRECTION: BC particles
        if (part_gpu->y[i] > grd_gpu->Ly){
            if (param_gpu->PERIODICY==true){ // PERIODIC
                part_gpu->y[i] = part_gpu->y[i] - grd_gpu->Ly;
            } else { // REFLECTING BC
                part_gpu->v[i] = -part_gpu->v[i];
                part_gpu->y[i] = 2*grd_gpu->Ly - part_gpu->y[i];
            }
        }
                                                                        
        if (part_gpu->y[i] < 0){
            if (param_gpu->PERIODICY==true){ // PERIODIC
                part_gpu->y[i] = part_gpu->y[i] + grd_gpu->Ly;
            } else { // REFLECTING BC
                part_gpu->v[i] = -part_gpu->v[i];
                part_gpu->y[i] = -part_gpu->y[i];
            }
        }
                                                                        
        // Z-DIRECTION: BC particles
        if (part_gpu->z[i] > grd_gpu->Lz){
            if (param_gpu->PERIODICZ==true){ // PERIODIC
                part_gpu->z[i] = part_gpu->z[i] - grd_gpu->Lz;
            } else { // REFLECTING BC
                part_gpu->w[i] = -part_gpu->w[i];
                part_gpu->z[i] = 2*grd_gpu->Lz - part_gpu->z[i];
            }
        }
                                                                        
        if (part_gpu->z[i] < 0){
            if (param_gpu->PERIODICZ==true){ // PERIODIC
                part_gpu->z[i] = part_gpu->z[i] + grd_gpu->Lz;
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
