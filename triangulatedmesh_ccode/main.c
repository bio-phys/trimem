// *****************************************************************************
//  CODE TO GET INFO ON THE FACES + VERTICES OF ANY TRIANGULATED MESH
//  MOTIVATION: Not be restricted to meshes available by Trimesh.icosphere
// *****************************************************************************

#include <stdlib.h> // ---> Library for random number generation
#include <stdio.h>
#include <math.h>

#include "PreprocessorDeclarations.h"
#include "DataStructures.h"
#include "Functions.h"

// *****************************************************************************
// MAIN
// *****************************************************************************
int main(int argc, const char * argv[]) {

    int p;
    char iiiii[500];
    FILE *o;
    lattice=1.75;

    N = atoi(argv[1]);

    // get faces of the triangles
    set_all();

    // -------------  save coordinates of vertices

    sprintf(iiiii,"mesh_coordinates_N_%d_.dat",N);
    o=fopen(iiiii,"w");

    for (p=1;p<=N;p++){
        // assumes that membrane beads have diameter 1.0
        fprintf(o,"%d %f %f %f\n",p-1, Elem[p].x,Elem[p].y,Elem[p].z);
    }

    fclose(o);

    // ------------- save faces triangles
    sprintf(iiiii,"mesh_faces_N_%d_.dat",N);
    o=fopen(iiiii,"w");

    for(p=1;p<=Ntri; p++){
      fprintf(o, "%d %d %d\n", Tr[p].v[0]-1, Tr[p].v[1]-1, Tr[p].v[2]-1);
    }

    fclose(o);

    return 0;
}
