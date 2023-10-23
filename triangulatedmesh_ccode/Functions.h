// *****************************************************************************
//
//  DYNAMICALLY TRIANGULATED MEMBRANE -- SARIC GROUP (ISTA)
//                 *** FUNCTION FILE ***
//
//  PLEASE NOTE that at the moment most
//  of the painters are commented out or out of use
//
//  For older and deprecated functions, please see
//  function cementery at the end of this file.
//
//  Read introduction in main.c file for further information.
//
// *****************************************************************************

#ifndef Functions_h
#define Functions_h

#include <stdio.h>
#include "DataStructures.h"
#include "PreprocessorDeclarations.h"

//..
////////////////////////////////////////////////////////////////////////////////
// *****************************************************************************
// FUNCTION DECLARATION
// *****************************************************************************
////////////////////////////////////////////////////////////////////////////////
// ..

// *****************************
// Functions for INITIALIZATION
// ****************************
void set_all(void);
void set_Elements(void);
void set_nb(void);
void set_triangles(void);

// **************************
// BASIC FUNCTION/OPERATIONS
// **************************
vec3D normale(int,int,int);
vec3D rotate(double , double ,double  ,double, double);
vec3D rotx(vec3D,double);
vec3D rotz(vec3D,double);
ANGOLO angle(double , double ,double);


// ..
////////////////////////////////////////////////////////////////////////////////
// *****************************************************************************
// FUNCTION DEFINITION
// *****************************************************************************
////////////////////////////////////////////////////////////////////////////////
//..

// -----------------------------
// FUNCTIONS FOR INITIALIZATION
// -----------------------------
void set_all(void){

    // maximum bond length for membrane beads
    cutoff=sqrt(3);
    cutoff2=cutoff*cutoff;

    // Geometry of the box set in 'PreprocessorDeclarations.h'
    // KEEP the '+2'
    S.sidex=(double)(LX+2);   // length X simulation box
    S.sidey=(double)(LY+2);   // length Y simulation box
    S.sidez=(double)(LZ+2);   // length Z simulation box

    S.side2x=S.sidex/2.0;
    S.side2y=S.sidey/2.0;
    S.side2z=S.sidez/2.0;
    S.Isidex=1.0/S.sidex;
    S.Isidey=1.0/S.sidey;
    S.Isidez=1.0/S.sidez;
    S.vol=S.sidex*S.sidey*S.sidez;  // volume of the simulation box
    S.Ivol=1./S.vol;

    // [ON] ALWAYS SET rv=rb
    S.rv=2.0;          // [ON] verlet minimum
    S.rb=S.rv;         // [ON] bond order minimum
    S.rc=1.78;         // [ON] cluster minimum
    S.rb2_2D=1.5*1.5;  // [ON] Bond Order Cut-OFF for 2D order parameter
    S.rv2=S.rv*S.rv;
    S.rb2=S.rb*S.rb;
    S.Vgap=0.5*(S.rv-1);  // [ON] put the largest instead of 1.

    S.Lcelx=(int)(S.sidex/S.rb);    // should give num cells along x
    S.Lcely=(int)(S.sidey/S.rb);    // should give num cells along y
    S.Lcelz=(int)(S.sidez/S.rb);    // should give num cells along z
    S.Ncel=S.Lcelx*S.Lcely*S.Lcelz; // total number of cells

    S.celsidex=S.sidex/S.Lcelx;
    S.celsidey=S.sidey/S.Lcely;
    S.celsidez=S.sidez/S.Lcelz;

    S.Icelsidex=1.0/S.celsidex;
    S.Icelsidey=1.0/S.celsidey;
    S.Icelsidez=1.0/S.celsidez;

    // --- allocate space for data structures (the +1 comes bc the zero-th index is not used in this code)
    Elem  =(PARTICLE *) malloc( (N+1)*sizeof(PARTICLE) );
    Tr    =(TRIANGLE *) malloc( (8*N+1)*sizeof(TRIANGLE) ); // why excess of triangles booked? (geometrically it should be around 2(N+1) triangles I think)
    Cella =(CELL *) malloc( (LX*LY*LZ+1)*sizeof(CELL) ); // total number of cells
    MEM   =(MEMBRANE *) malloc( (N+1)*sizeof(MEMBRANE) );

    set_Elements();
}
void set_Elements(void){

    // Function gets called when initial conditions file does not exist
    // This function produces said file by reading from
    // a file called icos_N_%d_Nnanop_%d_Sigma_%d_inorout_%d_seed_%d_.dat
    // There is a python file that produces these initial configurations
    //
    // Steps in the function:
    // 1. Read in particle coordinates from icos*
    // 2. Compute theta and phi angles
    // 3. Compute center of mass of the system
    // 4. Launch nb
    // 5. Set triangles
    // 6. Call painter2

    int i,j,k,p,t=0;
    double ssside,disp,vvv;
    double dx,dy,dz,d2;
    double Ex,Ey,Ez;
    double radius_membrane;
    double dummy_CX, dummy_CY, dummy_CZ;
    char to[500];
    FILE *in;

    // --- read file with initial configuration (use C code to produce it and Python code to edit it)
    sprintf(to,"icos_N_%d.dat",N);
    printf("Reading initial conditions from %s\n",to);
    in=fopen(to,"r");

    // --- read in line by line and save the coordinates of each particle (Nbeads and colloids)
    for (t=1;t<=N+Ncoll;t++){
        fscanf(in,"%d %lf %lf%lf\n",&Elem[t].type,&Elem[t].x,&Elem[t].y,&Elem[t].z);

        // slightly randomize initial positions? why?
        Elem[t].x+=.001*(.5-drand48());
        Elem[t].y+=.001*(.5-drand48());
        Elem[t].z+=.001*(.5-drand48());
    }

    // --- compute center of mass of the system (only for membrane)
    Centre.x=0.0;Centre.y=0.0;Centre.z=0.0;
    for (t=1;t<=N;t++){
        Centre.x+=Elem[t].x;
        Centre.y+=Elem[t].y;
        Centre.z+=Elem[t].z;
    }

    Centre.x/=(double)(N);
    Centre.y/=(double)(N);
    Centre.z/=(double)(N);

    printf("Center of mass of the system (%2.3f %2.3f %2.3f)\n",Centre.x,Centre.y,Centre.z);

    // --------------------- we must do this so that we can compute the triangles properly
    // displace first membrane to center
    dummy_CX = 0.0; dummy_CY = 0.0; dummy_CZ = 0.0;
    for(t=1; t<=N; t++){
        dummy_CX+= Elem[t].x;
        dummy_CY+= Elem[t].y;
        dummy_CZ+= Elem[t].z;
    }

    dummy_CX /= (double)(N);
    dummy_CY /= (double)(N);
    dummy_CZ /= (double)(N);

    for(t=1; t<=N; t++){
        MEM[t].x = Elem[t].x-dummy_CX;
        MEM[t].y = Elem[t].y-dummy_CY;
        MEM[t].z = Elem[t].z-dummy_CZ;
    }

    // since particles are always initialized in a sphere, enough to pick one particle
    S.rad = sqrt(MEM[1].x*MEM[1].x+MEM[1].y*MEM[1].y+MEM[1].z*MEM[1].z);

    set_nb();
    set_triangles();

}

void set_nb(void){

    // In this function we get the neighbours of beads in the membrane
    // and we set the bonds for the triangles therein

    int i,j,k,w,s,keepL,keepR;
    double dx,dy,dz,dr, dummy;
    double COS,SIN,tm_L,tm_R;
    double low=100000;
    vec3D rr[N+1];
    vec3D n,n_jk;
    ANGOLO cm;

    // --- iterate over membrane beads to get neighbours (triangular lattice first neighbours are 6)
    for (i=1;i<=N;i++){

        Elem[i].Ng=0;
        rr[i].x=0;
        rr[i].y=0;
        rr[i].z=0;

        for (j=1;j<=N;j++){

            if (i!=j){

                dx=Elem[i].x-Elem[j].x;
                dy=Elem[i].y-Elem[j].y;
                dz=Elem[i].z-Elem[j].z;

                dr=sqrt(dx*dx + dy*dy + dz*dz);

                // if these two membrane beads are close to each other
                if (dr<1.7){

                    // control to find out smallest distance
                    if (dr<low){
                        low=dr;
                    }

                    Elem[i].Ng++;               // neighbours of particle i -- neighbours in the lattice
                    Elem[i].ng[ Elem[i].Ng ]=j; // saving the specific neighbour of i, which is j

                }
            }
        }


        // if i has less than five neighbours or more than 7 --> error
        if ((Elem[i].Ng<5) || (Elem[i].Ng>7) ){
            printf("Problem: Triangular lattice seems to be poorly initialized because %d\n", Elem[i].Ng);
            exit(-1);
        }
    }

    // --- CONTROL: if two particles are overlapping, flag error
    if (low<1.0){
        printf("Problem: Min. dist bw two particles <1 =%f RAD=%f set it at least to %f\n",low,S.rad,S.rad/low);
        exit(-1);
    }


    // PLEASE KEEP IN MIND THAT THIS LOOP ONLY WORKS WELL IF THE CM OF THE VESICLE IS AT 0
    // This is why one has to use the MEM object
    // --- [0N] set bonds triagles -- k IS particle index
    for (k=1;k<=N;k++){

        // get position of membrane bead and transform into angles (used centered coords)
        cm=angle(MEM[k].x,MEM[k].y,MEM[k].z);
        // rotate vector so that it aligns with z-axis (sort of 'new base')
        rr[k]=rotate(MEM[k].x,MEM[k].y,MEM[k].z,cm.theta,cm.phi);

        // if it safe to ignore rr[k] -- it is simply (0, 0, radius of membrane)
        //printf("Components rr_k %lf %lf %lf\n", rr[k].x, rr[k].y, rr[k].z);

        // iterate around the neighbours of k
        for (w=1; w<=Elem[k].Ng; w++){
            j=Elem[k].ng[w];
            // rotate their coordinates by the same angles theta and phi to place them in the new axis
            rr[j]=rotate(MEM[j].x,MEM[j].y,MEM[j].z,cm.theta,cm.phi);
        }

        // for neighbours of k (called j)
        for (w=1;w<=Elem[k].Ng;w++){
            j=Elem[k].ng[w];

            dx=rr[j].x; // MMB removed rr[k].x --> it is 0
            dy=rr[j].y; // MMB removed rr[k].y --> it is 0

            dr=sqrt(dx*dx+dy*dy);
            n_jk.x=dx/dr;
            n_jk.y=dy/dr;

            tm_L=-2.0;
            tm_R=-2.0;

            // for neighbours of k that are not j (called s)
            for (i=1;i<=Elem[k].Ng;i++){

                if (i!=w){

                    s=Elem[k].ng[i];

                    dx=rr[s].x; // MMB removed rr[k].x --> it is 0
                    dy=rr[s].y; // MMB removed rr[k].y --> it is 0

                    dr=sqrt(dx*dx+dy*dy);
                    n.x=dx/dr;
                    n.y=dy/dr;

                    // cosine taken from scalar product
                    COS=n.x*n_jk.x + n.y*n_jk.y;

                    // sine taken from vector product
                    SIN=n_jk.x*n.y - n_jk.y*n.x;

                    if (SIN>0.0){ // [ON] s is "up" --> counterclockwise w.r.t. jk (right hand rule)
                        if (COS>tm_L){
                            keepL=i; // keep in mind we save the neighbor index ID, not the ID itself
                            tm_L=COS;
                        }
                    }

                    if (SIN<0.0){ // [ON] s is "down" --> clockwise w.r.t. jk
                        if (COS>tm_R){
                            keepR=i; // keep in mind we save the neighbor index ID, not the ID itself
                            tm_R=COS;
                        }
                    }
                }
            }

            // saving a different kind of neighbour - 'R' is right and 'L' is left?
            // for the line that connects particle k and the w-th neighbor, who is to the right, and who to the left
            Elem[k].ngR[w]=keepR;
            Elem[k].ngL[w]=keepL;
        }
    }
}
void set_triangles(void){

    int i,j,k,flag,t;
    int tp1,tp2,tp3;

    t=0;

    // for particles in membrane
    for (i=1;i<=N;i++){
        // for neighbours of particle i
        for (j=1;j<=Elem[i].Ng;j++){

            tp1=i;
            tp2=Elem[i].ng[j];  // getting a particle ID
            tp3=Elem[i].ngL[j]; // getting an index

            // [ON] tp3 tells me where is the Left neighbor in the list of site "i"
            tp3=Elem[i].ng[tp3]; // getting a particle ID

            //---------------
            // for triangles already added
            for (k=1;k<=t;k++){
                // [ON] verify that the triangle is not already taken
                flag=0;
                if (tp1==Tr[k].v[0] ||tp1==Tr[k].v[1] ||tp1==Tr[k].v[2]){
                    flag++;
                }
                if (tp2==Tr[k].v[0] ||tp2==Tr[k].v[1] ||tp2==Tr[k].v[2]){
                    flag++;
                }
                if (tp3==Tr[k].v[0] ||tp3==Tr[k].v[1] ||tp3==Tr[k].v[2]){
                    flag++;
                }
                if (flag==3){
                    // we have found a triangle 'k' with all vertices agreeing with our system
                    break;
                }
            }

            // saving the triangles (checks that has not been saved yet)
            if (flag!=3){
                t++;
                // add new triangle t with vertices v
                Tr[t].v[0]=tp1;
                Tr[t].v[1]=tp2;
                Tr[t].v[2]=tp3;
                Tr[t].Nv=3; // needed?
            }
            //---------------
        }
    }

    // counts and prints total number of triangles
    Ntri=t;
    printf("T=%d Ntri=%d\n",t,Ntri);
    printf("Expected Ntri=%d\n", 2*(N-2));

    // write down vertices coordinates and centroids in triangles membrane
    // MMB commenting them out
    //painter3();
    //painter4();

    // [ON] Now define the neighs of each triangle
    //      the neigh of v[1] must be the triangle t[1]
    //      that is opposite to it; (they must share 2 points = 1 line)

    for (i=1;i<=Ntri;i++){

        // [ON] Ngb 0 opposite to v[0]
        tp2=Tr[i].v[1];
        tp3=Tr[i].v[2];

        for (j=1;j<=Ntri;j++){
            if (i!=j){
                flag=0;
                if (tp2==Tr[j].v[0] ||tp2==Tr[j].v[1] ||tp2==Tr[j].v[2]){
                    flag++;
                }
                if (tp3==Tr[j].v[0] ||tp3==Tr[j].v[1] ||tp3==Tr[j].v[2]){
                    flag++;
                }
                if (flag==2){// accept
                    Tr[i].t[0]=j;
                    break;
                }
            }
        }

        // Ngb 1 opposite to v[1]
        tp1=Tr[i].v[0];
        tp3=Tr[i].v[2];

        for (j=1;j<=Ntri;j++){
            if (i!=j){
                flag=0;
                if (tp1==Tr[j].v[0] ||tp1==Tr[j].v[1] ||tp1==Tr[j].v[2]){
                    flag++;
                }
                if (tp3==Tr[j].v[0] ||tp3==Tr[j].v[1] ||tp3==Tr[j].v[2]){
                    flag++;
                }
                if (flag==2) {// accept
                    Tr[i].t[1]=j;
                    break;
                }
            }
        }

        // Ngb 2 opposite to v[2]
        tp1=Tr[i].v[0];
        tp2=Tr[i].v[1];

        for (j=1;j<=Ntri;j++){
            if (i!=j){
                flag=0;
                if (tp1==Tr[j].v[0] ||tp1==Tr[j].v[1] ||tp1==Tr[j].v[2]){
                    flag++;
                }
                if (tp2==Tr[j].v[0] ||tp2==Tr[j].v[1] ||tp2==Tr[j].v[2]){
                    flag++;
                }
                if (flag==2) {// accept
                    Tr[i].t[2]=j;
                    break;
                }
            }
        }

        Tr[i].Nt=3;
  }
}

// --------------------------------
// BASIC FUNCTIONS
// --------------------------------
vec3D normale(int o, int i,int j){

    // Function that computes the normal vector to the
    // plane defined by o, i and j

    double lg;
    vec3D a,b,n;

    // distance i, o
    a.x=Elem[i].x-Elem[o].x;
    a.y=Elem[i].y-Elem[o].y;
    a.z=Elem[i].z-Elem[o].z;

    // [ONLY PERIODIC BC THAT WERE HERE?]
    if (a.x>S.side2x)    a.x-=S.sidex;
    if (a.x<-S.side2x)   a.x+=S.sidex;
    if (a.y>S.side2y)    a.y-=S.sidey;
    if (a.y<-S.side2y)   a.y+=S.sidey;
    if (a.z>S.side2z)    a.z-=S.sidez;
    if (a.z<-S.side2z)   a.z+=S.sidez;

    // distance j, o
    b.x=Elem[j].x-Elem[o].x;
    b.y=Elem[j].y-Elem[o].y;
    b.z=Elem[j].z-Elem[o].z;

    // [ONLY PERIODIC BC THAT WERE HERE?]
    if (b.x>S.side2x)    b.x-=S.sidex;
    if (b.x<-S.side2x)   b.x+=S.sidex;
    if (b.y>S.side2y)    b.y-=S.sidey;
    if (b.y<-S.side2y)   b.y+=S.sidey;
    if (b.z>S.side2z)    b.z-=S.sidez;
    if (b.z<-S.side2z)   b.z+=S.sidez;

    // some form of vector product?
    n.x=a.y*b.z-a.z*b.y;
    n.y=a.z*b.x-a.x*b.z;
    n.z=a.x*b.y-a.y*b.x;

    // the length of the vector
    lg=sqrt(n.x*n.x+n.y*n.y+n.z*n.z);

    // normalize it
    n.x/=lg;
    n.y/=lg;
    n.z/=lg;

    return n;
}
vec3D rotate(double x,double y,double z,double theta, double phi){

    // Function that rotates x, y, z coordinates along angles theta and phi

    vec3D first,th;
    vec3D second;

    th.x=x;
    th.y=y;
    th.z=z;

    // rotation along z axis
    first=rotz(th,phi);

    // rotation along x axis
    second=rotx(first,theta);

    return second;
}
vec3D rotx(vec3D a,double Aangle){

    // Function that performs a rotation around z axis of angle Aangle

    vec3D rx;

    rx.x=a.x;
    rx.y=a.y*cos(Aangle)-a.z*sin(Aangle);
    rx.z=a.y*sin(Aangle)+a.z*cos(Aangle);

    return rx;
}
vec3D rotz(vec3D a,double Aangle){

    // Function that performs a rotation around z axis of angle Aangle

    vec3D rz;

    rz.x=a.x*cos(Aangle)-a.y*sin(Aangle);
    rz.y=a.x*sin(Aangle)+a.y*cos(Aangle);
    rz.z=a.z;

    return rz;
}
ANGOLO angle(double x,double y, double z){

    // Function gets coordinates of a membrane bead
    // and return the theta and phi angles of this
    // membrane bead on the surface of a sphere

    int i;
    double length,d0,sign;
    ANGOLO pp;
    vec3D nor;

    // --- get normalized position vector
    nor.x=x;
    nor.y=y;
    nor.z=z;

    length=sqrt(nor.x*nor.x+nor.y*nor.y+nor.z*nor.z);

    nor.x/=length;
    nor.y/=length;
    nor.z/=length;

    // --- get the theta angle
    // the acos function in C takes an argumen -1 <= x <= 1 and return the arc cosine
    // the if functions below make sure we are within the region where the function is defined
    if( nor.z>1.0-EPSILON ){
        nor.z=1.-EPSILON;
    }
    if ( nor.z<-1.+EPSILON ){
        nor.z=-1.+EPSILON;
    }

    pp.theta=acos(nor.z);

    // --- get the phi angle
    if (fabs(nor.y)<EPSILON && fabs(nor.x)<EPSILON ){
        pp.phi=0.;
        goto here;
    }
    if (fabs(nor.y)<EPSILON && fabs(nor.x)>EPSILON ){
        pp.phi=M_PI/2.;
    }

    // get the phi angle -- but it needs correction to see in what quadrant
    // THIS IS A BIT STRANGE BECAUSE THE TANGENT IS DEFINED THE OTHER WAY AROUND
    // MAY MAKE SENSE FOR THE KINDS OF ROTATIONS THAT COME AFTERWARDS
    pp.phi = atan(nor.x/nor.y);

    // not sure this 'here' makes a lot of sense
    // OR THAT THIS QUADRANT CORRECTIONS ARE OKAY
    // HAVE A LOOK AT WHAT ATAN DOES IN C
    here:{
        if (pp.phi!=0.){
            if (nor.x<0 && nor.y<0){
                pp.phi+=M_PI;
            }
            if (nor.x>0 && nor.y<0){
                pp.phi+=M_PI;
            }
        }
    }

    // returns angle with theta and phi
    return pp;
}

#endif /* Functions_h */
