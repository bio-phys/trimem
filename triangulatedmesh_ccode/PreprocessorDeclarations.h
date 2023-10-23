//
//  PreprocessorDeclarations.h
//  TriangulatedMonteCarloMembrane
//

#ifndef PreprocessorDeclarations_h
#define PreprocessorDeclarations_h

#define LX 40 // [ON] Number of cells per side of the simulation box
#define LY 40 // it must be an even number
#define LZ 40 // MMB: I think these are more the dimensions of the box

#define MCEQUILIBRATE 100
#define MCSTEPS 1000000
#define WRITE_CONF 500
#define EPSILON 1e-8

#define LENNARD_JONES 1
#define SQUARE_WELL 0
#define CHEMICAL_GRADIENT 0
#define HARMONIC_POTENTIAL 0

#define VOLSURFCONTROL 1

#endif /* PreprocessorDeclarations_h */
