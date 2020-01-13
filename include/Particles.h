#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

#define EPSILON 1.0e-4
#define TPB 256

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

//struct of the arrays in the particles struct
struct particles_a {

    FPpart* x; FPpart*  y; FPpart* z;
    FPpart* u; FPpart* v; FPpart* w;

};

enum copy_way{CPU_TO_GPU, GPU_TO_CPU};

void particle_allocate_gpu(struct particles* part, struct particles_a* part_gpu);

void particle_deallocate_gpu(struct particles_a* part_gpu);

void particle_copy(struct particles* part, struct particles_a* part_gpu, copy_way c);

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** deallocate */
void particle_deallocate(struct particles*);

/** particle mover */
void gpu_mover_PC(struct particles* part, struct grid* grd, struct parameters* param,
                            struct particles_a* part_gpu, struct EMfield_a field_gpu, struct grid_a grid_gpu);

__global__ void move_particle(int n_sub_cycles, int part_NiterMover, FPpart qom, struct grid grd, struct parameters param,
                                        struct particles_a part_gpu, struct EMfield_a field_gpu, struct grid_a grid_gpu);


/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

#endif
