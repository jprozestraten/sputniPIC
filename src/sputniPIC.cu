/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

// Verification
#include "Verification.h"

// Cuda profiler functions
#include <cuda_profiler_api.h>


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    

    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    

    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }
    
    printf("flag1\n");

    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);
    


    // Declaration of GPU variables for mover_PC
    parameters* param_gpu;
    grid* grd_gpu;
    EMfield* field_gpu;
    particles* part_gpu;

    //Allocation of pointers to struct for GPU variables for mover_PC
    cudaMalloc(&param_gpu, sizeof(parameters));
    cudaMalloc(&grd_gpu, sizeof(grd_gpu));
    cudaMalloc(&field_gpu, sizeof(field_gpu));
    cudaMalloc(&part_gpu, sizeof(part_gpu));


    //Data allocation of arrays in structs
    /*grid_allocate_gpu(&grd, grd_gpu);
    field_allocate_gpu(field_gpu, &grd);
    particle_allocate_gpu(part, part_gpu); */


    //Data copy of GPU variables for mover_PC
    cudaMemcpy(param_gpu, &(param), sizeof(parameters), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_gpu, &(grd), sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(field_gpu, &(field), sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu, &(part), sizeof(particles), cudaMemcpyHostToDevice);

    int TPB = 256;

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        // start profiler for mover_PC
        cudaProfilerStart();
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++)
        {
            cudaMemcpy(part_gpu, &(part), sizeof(particles), cudaMemcpyHostToDevice);
            gpu_mover_PC<<<(part_gpu->nop + TPB - 1)/TPB, TPB>>>(&part_gpu[is],field_gpu,grd_gpu,param_gpu);
            cudaMemcpy(&(part), part_gpu,  sizeof(particles), cudaMemcpyDeviceToHost);
        }
        eMover += (cpuSecond() - iMover); // stop timer for mover
        // stop profiler for mover_PC
        cudaProfilerStop();
        

        // start profiler for interpP2G
        cudaProfilerStart();

        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);

        // stop profiler for interpP2G
        cudaProfilerStop();

        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle

    cudaMemcpy(&(param), param_gpu, sizeof(parameters), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(grd), grd_gpu, sizeof(grid), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(field), field_gpu, sizeof(EMfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(part), part_gpu,  sizeof(particles), cudaMemcpyDeviceToHost);

    // Print if results are correct
    std::cout << std::endl;
    std::cout << "****************************" << std::endl;
    bool verification_result = verifyRhonet(&grd, &idn);
    std::cout << "   Results are correct: " << verification_result << std::endl;
    std::cout << "****************************" << std::endl;

    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    /*grid_deallocate_gpu(grd_gpu);
    field_deallocate_gpu(field_gpu);
    particle_deallocate_gpu(part_gpu); */


    cudaFree(param_gpu);
    cudaFree(grd_gpu);
    cudaFree(field_gpu);
    cudaFree(part_gpu);

    // exit
    return 0;
}

