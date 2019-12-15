#include <iostream>
#include <fstream>
#include <cmath>

#include "Verification.h"
#include "Grid.h"

/** verify results by checking idn->rhon against /testdata/rho_net, tol is relative */
bool verifyRhonet(struct grid *grd, struct interpDensNet *idn, FPinterp tol)
{
    // counter if all lines are correct
    int res = 0;

    // number from file
    FPinterp num_in;

    // input file
    std::ifstream fin("./testdata/rho_net.out");

    // get the number of nodes
    int nxn = grd->nxn;
    int nyn = grd->nyn;
    int nzn = grd->nzn;

    // checks all numbers, same loop as in RW_IO.cpp at end
    for (int k = 1; k < nzn - 2; k++) {
        for (int j = 1; j < nyn - 2; j++) {
            for (int i = 1; i < nxn - 2; i++) {
                fin >> num_in;

                // check if file reading went well
                if (fin.good()) {
                    // if numbers the same: increment counter, else false
                    if (fabs((num_in - idn->rhon[i][j][k]) / min(num_in, idn->rhon[i][j][k])) < tol) {
                        res++;
                    }
                    else {
                        return false;
                    }
                } 
                else {
                    std::cerr << "Error reading file: ./testdata/rho_net.out" << std::endl;
                    return false;
                }
            }
        }
    }

    // if res == number of lines in output file
    if (res == 32768) {
        return true;
    }
    else {
        return false;
    }
}
