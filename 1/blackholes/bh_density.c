#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/allvars.h"
#include "../main/proto.h"

#include "../domain/domain.h"


static int bh_density_evaluate(int target, int mode, int threadid);
static int bh_density_isactive(int n);

static MyFloat *BhNumNgb, *BhDhsmlDensityFactor;

/*! \brief Local data structure for collecting particle/cell data that is sent
 *         to other processors if needed. Type called data_in and static
 *         pointers DataIn and DataGet needed by generic_comm_helpers2.
 */
typedef struct
{
  MyDouble Pos[3];
  MyDouble Vel[3];
  MyFloat Hsml;
  int IsBh;
  int Firstnode;
} data_in;

static data_in *DataIn, *DataGet;

/*! \brief Routine that fills the relevant particle/cell data into the input
 *         structure defined above. Needed by generic_comm_helpers2.
 *
 *  \param[out] in Data structure to fill.
 *  \param[in] i Index of particle in P and SphP arrays.
 *  \param[in] firstnode First note of communication.
 *
 *  \return void
 */
static void particle2in(data_in *in, int i, int firstnode)
{
  in->Pos[0]        = PPB(i).Pos[0];
  in->Pos[1]        = PPB(i).Pos[1];
  in->Pos[2]        = PPB(i).Pos[2];
  in->Vel[0]        = PPB(i).Vel[0];
  in->Vel[1]        = PPB(i).Vel[1];
  in->Vel[2]        = PPB(i).Vel[2];
  in->Hsml          = BhP[i].Hsml;
  in->IsBh          = BhP[i].IsBh;
  in->Firstnode     = firstnode;
}  

/*! \brief Local data structure that holds results acquired on remote
 *         processors. Type called data_out and static pointers DataResult and
 *         DataOut needed by generic_comm_helpers2.
 */
typedef struct
{
  MyDouble DhsmlDensity;
  MyDouble Ngb;
  MyDouble Rho;
  MyDouble Mass;
  MyDouble MassFeed;
  integertime NgbMinStep;
#ifdef BONDI_ACCRETION
  MyDouble VelocityGas[3];
  MyDouble VelocityGasCircular[3];
  MyDouble InternalEnergyGas;
#endif
#ifdef INFALL_ACCRETION
  MyDouble Accretion;
#endif
} data_out;

static data_out *DataResult, *DataOut;

/*! \brief Routine to store or combine result data. Needed by
 *         generic_comm_helpers2.
 *
 *  \param[in] out Data to be moved to appropriate variables in global
 *  particle and cell data arrays (P, SphP,...)
 *  \param[in] i Index of particle in P and SphP arrays
 *  \param[in] mode Mode of function: local particles or information that was
 *  communicated from other tasks and has to be added locally?
 *
 *  \return void
 */
static void out2particle(data_out *out, int i, int mode)
{
  if(mode == MODE_LOCAL_PARTICLES) /* initial store */
    {
      BhDhsmlDensityFactor[i]          = out->DhsmlDensity;
      BhNumNgb[i]                      = out->Ngb;
      BhP[i].Density                   = out->Rho;
      BhP[i].NgbMass                   = out->Mass;
      BhP[i].NgbMassFeed               = out->MassFeed;
      BhP[i].NgbMinStep                = out->NgbMinStep;
#ifdef BONDI_ACCRETION
      BhP[i].VelocityGas[0]            = out->VelocityGas[0];
      BhP[i].VelocityGas[1]            = out->VelocityGas[1];
      BhP[i].VelocityGas[2]            = out->VelocityGas[2];
      BhP[i].VelocityGasCircular[0]    = out->VelocityGasCircular[0];
      BhP[i].VelocityGasCircular[1]    = out->VelocityGasCircular[1];
      BhP[i].VelocityGasCircular[2]    = out->VelocityGasCircular[2];
      BhP[i].InternalEnergyGas         = out->InternalEnergyGas;
#endif
#ifdef INFALL_ACCRETION
      BhP[i].Accretion                 = out->Accretion;
#endif
    }
  else /* combine */
    {
      BhDhsmlDensityFactor[i]          += out->DhsmlDensity;
      BhNumNgb[i]                      += out->Ngb;
      BhP[i].Density                   += out->Rho;
      BhP[i].NgbMass                   += out->Mass;
      BhP[i].NgbMassFeed               += out->MassFeed;
      if(out->NgbMinStep < BhP[i].NgbMinStep)
        BhP[i].NgbMinStep               = out->NgbMinStep;
#ifdef BONDI_ACCRETION
      BhP[i].VelocityGas[0]            += out->VelocityGas[0];
      BhP[i].VelocityGas[1]            += out->VelocityGas[1];
      BhP[i].VelocityGas[2]            += out->VelocityGas[2];
      BhP[i].VelocityGasCircular[0]    += out->VelocityGasCircular[0];
      BhP[i].VelocityGasCircular[1]    += out->VelocityGasCircular[1];
      BhP[i].VelocityGasCircular[2]    += out->VelocityGasCircular[2];
      BhP[i].InternalEnergyGas         += out->InternalEnergyGas;
#endif
#ifdef INFALL_ACCRETION
      BhP[i].Accretion                 += out->Accretion;
#endif
    }
}


#include "../utils/generic_comm_helpers2.h"

/*! \brief Routine that defines what to do with local particles.
 *
 *  Calls the *_evaluate function in MODE_LOCAL_PARTICLES.
 *
 *  \return void
 */
static void kernel_local(void)
{
  int i;

  {
    int j, threadid = get_thread_num();

    for(j = 0; j < NTask; j++)
      Thread[threadid].Exportflag[j] = -1;

    while(1)
      {
        if(Thread[threadid].ExportSpace < MinSpace)
          break;

        i = NextParticle++;

        if(i >= NumBh)
          break;

        if(bh_density_isactive(i))
          bh_density_evaluate(i, MODE_LOCAL_PARTICLES, threadid);
      }
  }
}

/*! \brief Routine that defines what to do with imported particles.
 *
 *  Calls the *_evaluate function in MODE_IMPORTED_PARTICLES.
 *
 *  \return void
 */
static void kernel_imported(void)
{
  /* now do the particles that were sent to us */
  int i, cnt = 0;
  {
    int threadid = get_thread_num();

    while(1)
      {
        i = cnt++;

        if(i >= Nimport)
          break;

        bh_density_evaluate(i, MODE_IMPORTED_PARTICLES, threadid);
      }
  }
}

/*! \brief Main function of SPH density calculation.
 *
 *  This function computes the local density for each active SPH particle and
 *  the number of weighted neighbors in the current smoothing radius. If a
 *  particle with its smoothing region is fully inside the local domain, it is
 *  not exported to the other processors. The function also detects particles
 *  that have a number of neighbors outside the allowed tolerance range. For
 *  these particles, the smoothing length is adjusted accordingly, and the
 *  computation is called again.
 *
 *  \return void
 */
void bh_density(void)
{
  MyFloat *Left, *Right;
  int i, npleft, iter = 0;
  long long ntot;
  double desnumngb, t0, t1;

  CPU_Step[CPU_MISC] += measure_time();

  BhNumNgb             = (MyFloat *)mymalloc("BhNumNgb", NumBh * sizeof(MyFloat));
  BhDhsmlDensityFactor = (MyFloat *)mymalloc("BhDhsmlDensityFactor", NumBh * sizeof(MyFloat));
  Left               = (MyFloat *)mymalloc("Left", NumBh * sizeof(MyFloat));
  Right              = (MyFloat *)mymalloc("Right", NumBh * sizeof(MyFloat));

  for(i = 0; i < NumBh; i++)
    {
      if(bh_density_isactive(i))
        {
          Left[i] = Right[i] = 0;
        }
    }

  generic_set_MaxNexport();

  desnumngb = All.BhDesNumNgb;

  /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
  do
    {
      t0 = second();

      generic_comm_pattern(NumBh, kernel_local, kernel_imported);

      for(i=0, npleft=0; i<NumBh; i++)
        {
          if(bh_density_isactive(i))
            {
              if(BhP[i].Density > 0)
                {
                  BhDhsmlDensityFactor[i] *= BhP[i].Hsml / (NUMDIMS * BhP[i].Density);
                  if(BhDhsmlDensityFactor[i] > -0.9) /* note: this would be -1 if only a single particle at zero lag is found */
                      BhDhsmlDensityFactor[i] = 1 / (1 + BhDhsmlDensityFactor[i]);
                  else
                      BhDhsmlDensityFactor[i] = 1;
                }
            } 
        

          if(BhNumNgb[i] < (desnumngb - All.BhMaxNumNgbDeviation) || BhNumNgb[i] > (desnumngb + All.BhMaxNumNgbDeviation))
          {
                  /* need to redo this particle */
            npleft++;

            if(Left[i] > 0 && Right[i] > 0)
              {
                if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
                  {
                        /* this one should be ok */
                    npleft--;
                    BhP[i].DensityFlag = -1; /* Mark as inactive */
                    continue;
                }
              } 

            if(BhNumNgb[i] < (desnumngb - All.BhMaxNumNgbDeviation))
              Left[i] = dmax(BhP[i].Hsml, Left[i]);
            else
              {
                if(Right[i] != 0)
                  {
                    if(BhP[i].Hsml < Right[i])
                        Right[i] = BhP[i].Hsml;
                  }
                    else
                        Right[i] = BhP[i].Hsml;
              }

            if(Right[i] > 0 && Left[i] > 0)
                BhP[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
            else
              {
                if(Right[i] == 0 && Left[i] == 0)
                    terminate("should not occur");

                if(Right[i] == 0 && Left[i] > 0)
                  {
                    BhP[i].Hsml *= 1.26;
                  }

                if(Right[i] > 0 && Left[i] == 0)
                  {
                    BhP[i].Hsml /= 1.26;
                  }
              }
          }
        else
             BhP[i].DensityFlag = -1; /* Mark as inactive */ 
        }

      sumup_large_ints(1, &npleft, &ntot);

      t1 = second();

      if(ntot > 0)
        {
          iter++;

          if(iter > 0)
            mpi_printf("BH_DENSITY: ngb iteration %3d: need to repeat for %12lld particles. (took %g sec)\n", iter, ntot,
                       timediff(t0, t1));

          if(iter > MAXITER)
            terminate("failed to converge in neighbour iteration in bh_density()\n");
        }
    }
  while(ntot > 0);


  myfree(Right);
  myfree(Left);
  myfree(BhDhsmlDensityFactor);
  myfree(BhNumNgb);

  /* mark as active again */
for(i = 0; i < NumBh; i++)
    {
     BhP[i].DensityFlag = 1;
    }
  /* collect some timing information */
  CPU_Step[CPU_INIT] += measure_time();
}

/*! \brief Inner function of the SPH density calculation
 *
 *  This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 *
 *  \param[in] target Index of particle in local data/import buffer.
 *  \param[in] mode Mode in which function is called (local or impored data).
 *  \param[in] threadid ID of local thread.
 *
 *  \return 0
 */
static int bh_density_evaluate(int target, int mode, int threadid)
{
  int j, n, isbh;
  int numngb, numnodes, *firstnode;
  double h, h2, hinv, hinv3, hinv4;
  double rho;
  double wk, dwk;
  double dx, dy, dz, r, r2, u, mass_j;
  MyFloat weighted_numngb;
  MyFloat dhsmlrho;
  MyDouble *pos;
  MyDouble mass, mass_feed;
  integertime ngb_min_step;
  int bin = TIMEBINS;

  data_in local, *target_data;
  data_out out;

  if(mode == MODE_LOCAL_PARTICLES)
    {
      particle2in(&local, target, 0);
      target_data = &local;

      numnodes  = 1;
      firstnode = NULL;
    }
  else
    {
      target_data = &DataGet[target];

      generic_get_numnodes(target, &numnodes, &firstnode);
    }

  pos  = target_data->Pos;
  h    = target_data->Hsml;
  isbh = target_data->IsBh;

#ifdef BONDI_ACCRETION
  MyDouble *vel;
  vel  = target_data->Vel;
  double dvx, dvy, dvz, rho_j;
  MyDouble internal_energy_gas = 0;
  MyDouble velocity_gas[3], velocity_gas_circular[3];
  velocity_gas[0] = velocity_gas[1] = velocity_gas[2] = 0;
  velocity_gas_circular[0] = velocity_gas_circular[1] = velocity_gas_circular[2] = 0;
#endif
#ifdef INFALL_ACCRETION
  double accretion = 0;
#endif 

  h2   = h * h;
  hinv = 1.0 / h;
#ifndef TWODIMS
  hinv3 = hinv * hinv * hinv;
#else  /* #ifndef  TWODIMS */
  hinv3 = hinv * hinv / boxSize_Z;
#endif /* #ifndef  TWODIMS #else */
  hinv4 = hinv3 * hinv;

  numngb = 0;
  rho = weighted_numngb = dhsmlrho = 0;
  mass = mass_feed = 0;


/*jet axis and opening angle*/    

/*positive and negative jet axes (no need to be normalized) */
  double pos_x_axis[3] = {1, 0, 0};
  double neg_x_axis[3] = {-1, 0, 0};      
/*jet angle*/
  double theta = M_PI/4;
  double vx, vy, vz, pos_x_angle, neg_x_angle; 

  int nfound = ngb_treefind_variable_threads(pos, h, target, mode, threadid, numnodes, firstnode);

  for(n = 0; n < nfound; n++)
    {
      j = Thread[threadid].Ngblist[n];

/*compute bh->cell position (and velocity for bhs later) vectors: posBhP-posSphP (while velSphP-velBhP)*/
      dx = pos[0] - P[j].Pos[0];
      dy = pos[1] - P[j].Pos[1];
      dz = pos[2] - P[j].Pos[2];

/* now find the closest image in the given box size */
#ifndef REFLECTIVE_X
      if(dx > boxHalf_X)
        dx -= boxSize_X;
      if(dx < -boxHalf_X)
        dx += boxSize_X;
#endif /* #ifndef REFLECTIVE_X */

#ifndef REFLECTIVE_Y
      if(dy > boxHalf_Y)
        dy -= boxSize_Y;
      if(dy < -boxHalf_Y)
        dy += boxSize_Y;
#endif /* #ifndef REFLECTIVE_Y */

#ifndef REFLECTIVE_Z
      if(dz > boxHalf_Z)
        dz -= boxSize_Z;
      if(dz < -boxHalf_Z)
        dz += boxSize_Z;
#endif /* #ifndef REFLECTIVE_Z */
      r2 = dx * dx + dy * dy + dz * dz;

      if(r2 < h2)
        {
          numngb++;

          r = sqrt(r2);

          u = r * hinv;

          kernel(u, hinv3, hinv4, &wk, &dwk);

          mass_j = P[j].Mass;

/*compute bh density*/
          rho += FLT(mass_j * wk);

          weighted_numngb += FLT(NORM_COEFF * wk / hinv3); /* 4.0/3 * PI = 4.188790204786 */

          dhsmlrho += FLT(-mass_j * (NUMDIMS * hinv * wk + u * dwk));

/*compute the min hydro step for neighbors*/     
          if(bin > P[j].TimeBinHydro)
            bin = P[j].TimeBinHydro;

/*compute the bh-ngb-mass*/
          mass += mass_j;

          if(isbh)
            {
#ifdef BONDI_ACCRETION
/*compute relative velocities, relative specific angular momenta and internal energy of gas*/
              dvx = P[j].Vel[0] - vel[0]; 
              dvy = P[j].Vel[1] - vel[1]; 
              dvz = P[j].Vel[2] - vel[2]; 

              if(SphP[j].Density > 0)
                rho_j  = SphP[j].Density;
              else
                rho_j = 1;

              velocity_gas[0] += dvx*mass_j/rho_j*wk;
              velocity_gas[1] += dvy*mass_j/rho_j*wk;
              velocity_gas[2] += dvz*mass_j/rho_j*wk;

              velocity_gas_circular[0] -= (dy * dvz - dz * dvy)*mass_j/rho_j*wk;
              velocity_gas_circular[1] -= (dz * dvx - dx * dvz)*mass_j/rho_j*wk;
              velocity_gas_circular[2] -= (dx * dvy - dy * dvx)*mass_j/rho_j*wk;

              internal_energy_gas += SphP[j].Utherm*mass_j/rho_j*wk;
#endif
#ifdef INFALL_ACCRETION
              /*if(r2 < 0.00001)
                {
                  if(SphP[j].MassDrain >= 0)
                    {
                      
                      accretion += 0.9*mass_j;
                      
                    }
                }*/
             
              P[j].Mass -= P[j].Mass * exp(-r2);
              accretion += P[j].Mass * exp(-r2);
#endif
              if(All.JetFeedback)
                {
/*double cone jet setup*/    
  
/*calculate vector to cone vertex*/
                  vx = -dx; // x-component of the vector from the vertex to the point
                  vy = -dy; // y-component of the vector from the vertex to the point
                  vz = -dz; // z-component of the vector from the vertex to the point
/*calculate angles*/    
                  pos_x_angle = acos((vx*pos_x_axis[0] + vy*pos_x_axis[1] + vz*pos_x_axis[2]) / 
                  (sqrt(pow(vx, 2) + pow(vy, 2) + pow(vz, 2)) * sqrt(pow(pos_x_axis[0], 2) + pow(pos_x_axis[1], 2) +  pow(pos_x_axis[2], 2))));
                  neg_x_angle = acos((vx*neg_x_axis[0] + vy*neg_x_axis[1] + vz*neg_x_axis[2]) / 
                  (sqrt(pow(vx, 2) + pow(vy, 2) + pow(vz, 2)) * sqrt(pow(neg_x_axis[0], 2) + pow(neg_x_axis[1], 2) + pow(neg_x_axis[2], 2))));
/*check if particle is inside the cone*/ 
                  if((pos_x_angle <= theta) || (neg_x_angle <= theta))
                    mass_feed += P[j].Mass;
                }
            }  
        }
    }

/*compute bh timestep based on min ngb timestep*/
  if(bin == 0)
    ngb_min_step = 0;
  else
    ngb_min_step   = (((integertime)1) << bin);
  
  out.DhsmlDensity            = dhsmlrho;
  out.Ngb                     = weighted_numngb;
  out.Rho                     = rho;
  out.Mass                    = mass;
  out.MassFeed                = mass_feed;
  out.NgbMinStep              = ngb_min_step;
#ifdef BONDI_ACCRETION
  out.VelocityGas[0]          = velocity_gas[0];
  out.VelocityGas[1]          = velocity_gas[1];
  out.VelocityGas[2]          = velocity_gas[2];
  out.VelocityGasCircular[0]  = velocity_gas_circular[0];
  out.VelocityGasCircular[1]  = velocity_gas_circular[1];
  out.VelocityGasCircular[2]  = velocity_gas_circular[2];
  out.InternalEnergyGas       = internal_energy_gas;
#endif
#ifdef INFALL_ACCRETION
  out.Accretion               = accretion;
#endif
  /* now collect the result at the right place */
  if(mode == MODE_LOCAL_PARTICLES)
    out2particle(&out, target, MODE_LOCAL_PARTICLES);
  else
    DataResult[target] = out;

  return 0;
}

/* \brief Determines if a BhP is active in current timestep.
 *
 *  \param[in] n Index of BhP in Particle array
 *
 *  \return 1: BhP active; 0: BhP not active.
 */
int bh_density_isactive(int n)
{
  if(BhP[n].DensityFlag < 0)
    return 0;

  return 1;
}
