#include <string>
#include "Parameters.h"

/*

void param_allocate_gpu(struct parameters* param_gpu)
{
    cudaMalloc(&(param_gpu->npcelx), NS_MAX*sizeof(int));
    cudaMalloc(&(param_gpu->npcely), NS_MAX*sizeof(int));
    cudaMalloc(&(param_gpu->npcelz), NS_MAX*sizeof(int));
    cudaMalloc(&(param_gpu->np), NS_MAX*sizeof(long));
    cudaMalloc(&(param_gpu->npMax), NS_MAX*sizeof(long));
    cudaMalloc(&(param_gpu->qom), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->rhoINIT), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->uth), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->vth), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->wth), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->u0), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->v0), NS_MAX*sizeof(double));
    cudaMalloc(&(param_gpu->w0), NS_MAX*sizeof(double));
}
*/



void init_param_gpu(struct parameters* param, struct parameters* param_gpu)
{
	/** light speed */
    param_gpu->c = param->c;
    /** 4  pi */
    param_gpu->fourpi = param->fourpi;
    /** time step */
    param_gpu->dt = param->dt;
    /** decentering param_gpueter */
    param_gpu->th = param->th;
    
    /** number of time cycles */
    param_gpu->ncycles = param->ncycles;
    /** mover predictor correcto iteration */
    param_gpu->NiterMover = param->NiterMover;
    /** number of particle of subcycles in the mover */
    param_gpu->n_sub_cycles = param->n_sub_cycles;
    
    /** simulation box length - X direction   */
    param_gpu->Lx = param->Lx;
    /** simulation box length - Y direction   */
    param_gpu->Ly = param->Ly;
    /** simulation box length - Z direction   */
    param_gpu->Lz = param->Lz;
    /** number of cells - X direction        */
    param_gpu->nxc = param->nxc;
    /** number of cells - Y direction        */
    param_gpu->nyc = param->nyc;
    /** number of cells - Z direction        */
    param_gpu->nzc = param->nzc;
    /** object center X, e.g. planet or comet   */
    param_gpu->x_center = param->x_center;
    /** object center Y, e.g. planet or comet   */
    param_gpu->y_center = param->y_center;
    /** object center Z, e.g. planet or comet   */
    param_gpu->z_center = param->z_center;
    /** object size - assuming a cubic box or sphere  */
    param_gpu->L_square = param->L_square;
    
    
    /** number of actual species */
    param_gpu->ns = param->ns;
    
    // This for maximum NS_MAX species. To have more increase the array size in NS_MAX
    /** number of particles per cell - X direction */
    cudaMemcpy(param_gpu->npcelx, param->npcelx, NS_MAX*sizeof(int), cudaMemcpyHostToDevice);
    /** number of particles per cell - Y direction */
    cudaMemcpy(param_gpu->npcely, param->npcely, NS_MAX*sizeof(int), cudaMemcpyHostToDevice);
    /** number of particles per cell - Z direction */
    cudaMemcpy(param_gpu->npcelz, param->npcelz, NS_MAX*sizeof(int), cudaMemcpyHostToDevice);
    /** number of particles array for different species */
    cudaMemcpy(param_gpu->np, param->np, NS_MAX*sizeof(long), cudaMemcpyHostToDevice);
    /** maximum number of particles array for different species */
    cudaMemcpy(param_gpu->npMax, param->npMax, NS_MAX*sizeof(long), cudaMemcpyHostToDevice);
    /** max number of particles */
    param_gpu->NpMaxNpRatio = param->NpMaxNpRatio;
    /** charge to mass ratio array for different species */
    cudaMemcpy(param_gpu->qom, param->qom, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** charge to mass ratio array for different species */
    cudaMemcpy(param_gpu->rhoINIT, param->rhoINIT, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** thermal velocity  - Direction X  */
    cudaMemcpy(param_gpu->uth, param->uth, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** thermal velocity  - Direction Y  */
    cudaMemcpy(param_gpu->vth, param->vth, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** thermal velocity  - Direction Z  */
    cudaMemcpy(param_gpu->wth, param->wth, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** Drift velocity - Direction X     */
    cudaMemcpy(param_gpu->u0, param->u0, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** Drift velocity - Direction Y    */
    cudaMemcpy(param_gpu->v0, param->v0, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    /** Drift velocity - Direction Z     */
    cudaMemcpy(param_gpu->w0, param->w0, NS_MAX*sizeof(double), cudaMemcpyHostToDevice);
    
    
    
    /** Boundary Condition: Periodicity **/
    // here you have to set the topology for the fields
    /** Periodicity for fields X **/

    param_gpu->PERIODICX = param->PERIODICX;
    /** Periodicity for fields Y **/
    param_gpu->PERIODICY = param->PERIODICY;
    /** Periodicity for fields Z **/
    param_gpu->PERIODICZ = param->PERIODICZ;
    /** Periodicity for Particles X **/
    param_gpu->PERIODICX_P = param->PERIODICX_P;
    /** Periodicity for Particles Y **/
    param_gpu->PERIODICY_P = param->PERIODICY_P;
    /** Periodicity for Particles Y **/
    param_gpu->PERIODICZ_P = param->PERIODICZ_P;
    
    
    /** Boundary condition on particles
     0 = exit
     1 = perfect mirror
     2 = riemission
     */
    /** Boundary Condition Particles: FaceXright */
    param_gpu->bcPfaceXright = param->bcPfaceXright;
    /** Boundary Condition Particles: FaceXleft */
    param_gpu->bcPfaceXleft = param->bcPfaceXleft;
    /** Boundary Condition Particles: FaceYright */
    param_gpu->bcPfaceYright = param->bcPfaceYright;
    /** Boundary Condition Particles: FaceYleft */
    param_gpu->bcPfaceYleft = param->bcPfaceYleft;
    /** Boundary Condition Particles: FaceYright */
    param_gpu->bcPfaceZright = param->bcPfaceZright;
    /** Boundary Condition Particles: FaceYleft */
    param_gpu->bcPfaceZleft = param->bcPfaceZleft;
    
    
    /** Field Boundary Condition
     0 =  Dirichlet Boundary Condition: specifies the valueto take pn the boundary of the domain
     1 =  Neumann Boundary Condition: specifies the value of derivative to take on the boundary of the domain
     2 =  Periodic Condition
     */
    /** Boundary Condition Electrostatic Potential: FaceXright */
    param_gpu->bcPHIfaceXright = param->bcPHIfaceXright;
    /** Boundary Condition Electrostatic Potential:FaceXleft */
    param_gpu->bcPHIfaceXleft = param->bcPHIfaceXleft;
    /** Boundary Condition Electrostatic Potential:FaceYright */
    param_gpu->bcPHIfaceYright = param->bcPHIfaceYright;
    /** Boundary Condition Electrostatic Potential:FaceYleft */
    param_gpu->bcPHIfaceYleft = param->bcPHIfaceYleft;
    /** Boundary Condition Electrostatic Potential:FaceZright */
    param_gpu->bcPHIfaceZright = param->bcPHIfaceZright;
    /** Boundary Condition Electrostatic Potential:FaceZleft */
    param_gpu->bcPHIfaceZleft = param->bcPHIfaceZleft;
    
    /** Boundary Condition EM Field: FaceXright */
    param_gpu->bcEMfaceXright = param->bcEMfaceXright;
    /** Boundary Condition EM Field: FaceXleft */
    param_gpu->bcEMfaceXleft = param->bcEMfaceXleft;
    /** Boundary Condition EM Field: FaceYright */
    param_gpu->bcEMfaceYright = param->bcEMfaceYright;
    /** Boundary Condition EM Field: FaceYleft */
    param_gpu->bcEMfaceYleft = param->bcEMfaceYleft;
    /** Boundary Condition EM Field: FaceZright */
    param_gpu->bcEMfaceZright = param->bcEMfaceZright;
    /** Boundary Condition EM Field: FaceZleft */
    param_gpu->bcEMfaceZleft = param->bcEMfaceZleft;
    
    /** velocity of the injection from the wall */
    param_gpu->Vinj = param->Vinj;
    
    
    /** Initial Condition*/
    /** current sheet thickness */
    param_gpu->delta = param->delta;
    /* Initial B0x */
    param_gpu->B0x = param->B0x;
    /* Initial B0y */
    param_gpu->B0y = param->B0y;
    /* Initial B0y */
    param_gpu->B0z = param->B0z;
    /** Number of waves present in the system: used for turbulence studies */
    param_gpu->Nwaves = param->Nwaves;    
    /** Perturbation amplitude: used for turbulence studies */
    param_gpu->dBoB0 = param->dBoB0;
    /** pitch angle */
    param_gpu->pitch_angle = param->pitch_angle;
    /** energy of the particle */
    param_gpu->energy = param->energy;
    
    /** Smoothing quantities */
    param_gpu->SmoothON = param->SmoothON;
    /** Smoothing value*/
    param_gpu->SmoothValue = param->SmoothValue;
    /** Ntimes: smoothing is applied */
    param_gpu->SmoothTimes = param->SmoothTimes;
    
    
    /** boolean value for verbose results */
    param_gpu->verbose = param->verbose;
    /** RESTART */
    param_gpu->RESTART = param->RESTART;
    
    
    /** Poisson Correction */
    param_gpu->PoissonCorrection = param->PoissonCorrection;
    /** CG solver stopping criterium tolerance */
    param_gpu->CGtol = param->CGtol;
    /** GMRES solver stopping criterium tolerance */
    param_gpu->GMREStol = param->GMREStol;
    
    /** needed if restart */
    param_gpu->first_cycle_n = param->first_cycle_n;
    
    /** Output for field */
    param_gpu->FieldOutputCycle = param->FieldOutputCycle;
    /** Output for particles */
    param_gpu->ParticlesOutputCycle = param->ParticlesOutputCycle;
    /** restart cycle */
    param_gpu->RestartOutputCycle = param->RestartOutputCycle;
    /** Output for diagnostics */
    param_gpu->DiagnosticsOutputCycle = param->DiagnosticsOutputCycle;
    
    /** inputfile */
    //strcpy(param_gpu->inputfile, param->inputfile);
    /** SaveDirName     */
    //strcpy(param_gpu->SaveDirName, param->SaveDirName);
    /** RestartDirName     */
    //strcpy(param_gpu->RestartDirName, param->RestartDirName);
    /** name of the file with wave amplitude and phases */
    //strcpy(param_gpu->WaveFile, param->WaveFile);
}