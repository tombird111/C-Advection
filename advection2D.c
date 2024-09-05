/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  /* xmax and ymax changed to 30 */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value
  
  /* Parameters for the Gaussian initial conditions */
  /* Centre changed to x = 3.0, y = 15.0 */
  /* Wudth changed to x = 1.0, y = 5.0 */
  const float x0=3.0;                    // Centre(x)
  const float y0=15.0;                   // Centre(y)
  const float sigmax=1.0;               // Width(x)
  const float sigmay=5.0;               // Width(y)
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary
  
  /* Wind properties */
  const float ustar = 0.2; // Friction velocity
  const float z0 = 1.0; // Roughness length
  const float k = 0.41; // Von Karmans constant
  
  /* Time stepping parameters */
  /* Timesteps changed from 1500 to 800*/
  /* To obey Courant-Friedrichs-Lewy condition, the CFL number should be changed */
  /* Maximum distance x can advect in a second = vxmax */
  /* CFL should be between 0 and 1, and be inversely proportional to vxmax */
  const float vxmax = (ustar/k) * log(ymax/z0); //Calculate the maximum amount x can change in a second
  const float CFL=(1/vxmax);   // CFL number 
  const int nsteps=800; // Number of time steps

  /* Velocity */
  /* x and y velocities changed to 1.0 and 0.0 respectively */
  const float velx=1.0; // Velocity in x direction
  const float vely=0.0; // Velocity in y direction
  
  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  float x2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);

  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(velx) / dx) + (fabs(vely) / dy) );
  
  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */
  /* These two loops can be parallelised as they operate based on constant variables and single loop steps */
  #pragma omp parrallel for default(none) shared(x, dx)
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }
  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  #pragma omp parallel for default(none) shared(y, dy)
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }
  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */
  /* This loop can be parallelised twice, for both i and j */
  /* This is because x2 and y2 are used privately in each case, u is written to individual cells, and x0/y0 are constants */
  #pragma omp parallel for collapse(2) default(none) shared(u, x, y) private(x2, y2) 
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      x2      = (x[i]-x0) * (x[i]-x0);
      y2      = (y[j]-y0) * (y[j]-y0);
      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
    }
  }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  /* This loop should not be parallelised as it is writing to a file, and we want the file to be written to in order */
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
  
  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */
  /* This loop cannot be parallelised due to the large amount of flow dependencies within */
  for (int m=0; m<nsteps; m++){
    
    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
    /* This loop may be parallelised as it is writing to individual border values */
    #pragma omp parallel for default(none) shared(u)
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
    /* This may be parallelised similarly to loop 6 */
    #pragma omp parallel for default(none) shared(u)
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    

    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
    /* These loops may be parallelised as the two loops write to a separate array */
    #pragma omp parallel for collapse(2) default(none) shared(u, dudt, dx, dy, y)
    for (int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
        float height = y[j];
        float newvelx = 0.0;
        if (height <= z0) { // If the height is below the roughness length, calculate with x velocity = 0
          dudt[i][j] = -newvelx * (u[i][j] - u[i-1][j]) / dx
	              - vely * (u[i][j] - u[i][j-1]) / dy;
        } else { // If the height is above the roughness length, calculate using the equation
          newvelx = (ustar/k) * (log(height/z0));
	  dudt[i][j] = -newvelx * (u[i][j] - u[i-1][j]) / dx
	              - vely * (u[i][j] - u[i][j-1]) / dy;
        }
      }
    }
    
    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    /* These loops may be parallelised as they are dependent on private, unique sets of co-ordinates */
    #pragma omp parallel for collapse(2) default(none) shared(u, dudt, dt)
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
	u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
    
  } // time loop
  
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  /* This loop should not be parallelised as it is writing to a file, and we want the file to be written to in order */
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  FILE *avgfile;
  avgfile = fopen("avgfile.dat", "w");
  float sum = 0.0;
  for (int i=1; i<NY+1; i++){ //For every x value, ignoring boundaries with 0 and NY + 1
    for (int j=1; j<NX+1; j++){ //For every y value, ignoring boundaries 0 and NY + 1
      sum = sum + u[i][j]; //Add the u at each y value related to that x value
    }
    fprintf(avgfile, "%g %g\n", y[i], (sum/(NY+1.0))); //Print the averages to the file
    sum = 0.0;
  }
  fclose(avgfile);
  
  return 0;
}

/* End of file ******************************************************/
