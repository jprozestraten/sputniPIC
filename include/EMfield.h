#ifndef EMFIELD_H
#define EMFIELD_H

#include "Alloc.h"
#include "Grid.h"


/** structure with field information */
struct EMfield {
    // field arrays: 4D arrays
    
    /* Electric field defined on nodes: last index is component */
    FPfield*** Ex;
    FPfield* Ex_flat;
    FPfield*** Ey;
    FPfield* Ey_flat;
    FPfield*** Ez;
    FPfield* Ez_flat;
    /* Magnetic field defined on nodes: last index is component */
    FPfield*** Bxn;
    FPfield* Bxn_flat;
    FPfield*** Byn;
    FPfield* Byn_flat;
    FPfield*** Bzn;
    FPfield* Bzn_flat;
    
    
};

struct EMfield_a {
        /* Electric field defined on nodes: last index is component */
    FPfield* Ex_flat;
    FPfield* Ey_flat;
    FPfield* Ez_flat;
    /* Magnetic field defined on nodes: last index is component */
    FPfield* Bxn_flat;
    FPfield* Byn_flat;
    FPfield* Bzn_flat;
    
};


void field_allocate_gpu(struct EMfield_a* field_gpu, struct grid* grd);

void field_deallocate_gpu(struct EMfield_a* field_gpu);

void field_copy(struct EMfield* field, struct EMfield_a field_gpu, struct grid* grd);

/** allocate electric and magnetic field */
void field_allocate(struct grid*, struct EMfield*);

/** deallocate electric and magnetic field */
void field_deallocate(struct grid*, struct EMfield*);

#endif
