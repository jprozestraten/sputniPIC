#ifndef VERIFICATION_H
#define VERIFICATION_H

#include "PrecisionTypes.h"

/** verify results by checking idn->rhon against /testdata/rho_net, tol is relative */
bool verifyRhonet(struct grid*, struct interpDensNet*, FPinterp tol = 1e-5);

#endif
