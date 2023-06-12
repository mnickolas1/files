#include <gsl/gsl_math.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/allvars.h"
#include "../main/proto.h"

#include "../domain/domain.h"


static int bh_ngb_feedback_evaluate(int target, int mode, int threadid);

/*! \brief Local data structure for collecting particle/cell data that is sent
 *         to other processors if needed. Type called data_in and static
 *         pointers DataIn and DataGet needed by generic_comm_helpers2.
 */
typedef struct
{
  MyDouble Pos[3];
  MyFloat Hsml;
  int Bin;
  int IsBh;
  MyDouble BhRho;
  MyDouble BhMass;
  MyDouble NgbMass;
  MyDouble NgbMassFeed;
#ifdef BONDI_ACCRETION
  MyDouble AccretionRate;
  MyDouble MassToDrain;
#endif
#ifdef INFALL_ACCRETION
  MyDouble Accretion;
#endif
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
  in->Hsml          = BhP[i].Hsml;
  in->Bin           = BhP[i].TimeBinBh;
  in->IsBh          = BhP[i].IsBh;
  in->BhRho         = BhP[i].Density;
  in->BhMass        = PPB(i).Mass;
  in->NgbMass       = BhP[i].NgbMass;
  in->NgbMassFeed   = BhP[i].NgbMassFeed;
#ifdef BONDI_ACCRETION 
  in->AccretionRate = BhP[i].AccretionRate;
  in->MassToDrain   = BhP[i].MassToDrain;
#endif
#ifdef INFALL_ACCRETION
  in->Accretion     = BhP[i].Accretion;
#endif
  in->Firstnode     = firstnode;
}

/*! \brief Local data structure that holds results acquired on remote
 *         processors. Type called data_out and static pointers DataResult and
 *         DataOut needed by generic_comm_helpers2.*/

typedef struct
{

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
 *  \return void*/
 
static void out2particle(data_out *out, int i, int mode)
{
  if(mode == MODE_LOCAL_PARTICLES) /* initial store */
    {
      return;
    }
  else /* combine */
    {
      return;
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
  int idx;

  {
    int j, threadid = get_thread_num();

    for(j = 0; j < NTask; j++)
      Thread[threadid].Exportflag[j] = -1;

    while(1)
      {
        if(Thread[threadid].ExportSpace < MinSpace)
          break;

        idx = NextParticle++;

        if(idx >= TimeBinsBh.NActiveParticles)
          break;

        int i = TimeBinsBh.ActiveParticleList[idx];
        if(i < 0)
          continue;
        bh_ngb_feedback_evaluate(i, MODE_LOCAL_PARTICLES, threadid);
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

        bh_ngb_feedback_evaluate(i, MODE_IMPORTED_PARTICLES, threadid);
      }
  }
}

void bh_ngb_feedback(void)
{
  generic_set_MaxNexport();

  generic_comm_pattern(TimeBinsBh.NActiveParticles, kernel_local, kernel_imported);
}

static int bh_ngb_feedback_evaluate(int target, int mode, int threadid)
{
  int j, n, bin, isbh;
  int numnodes, *firstnode;
  double h, h2, hinv, hinv3, hinv4, wk, dwk;
  double dx, dy, dz, r, r2, u;
  double dt; //dtime;
  MyDouble bh_mass, ngbmass, ngbmass_feed; 
  MyDouble *pos, bh_rho, energyfeed;

  data_in local, *target_data;
  /*data_out out;*/

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

  pos            = target_data->Pos;
  h              = target_data->Hsml;
  isbh           = target_data->IsBh;
  bin            = target_data->Bin;
  bh_rho         = target_data->BhRho;
  bh_mass        = target_data->BhMass;
  ngbmass        = target_data->NgbMass;
  ngbmass_feed   = target_data->NgbMassFeed;

#ifdef BONDI_ACCRETION
  MyDouble accretion_rate, mass_to_drain;
  accretion_rate = target_data->AccretionRate;
  mass_to_drain  = target_data->MassToDrain; 
#endif
#ifdef INFALL_ACCRETION
  MyDouble accretion;
  accretion      = target_data->Accretion;
#endif

  h2   = h * h;
  hinv = 1.0 / h;
#ifndef TWODIMS
  hinv3 = hinv * hinv * hinv;
#else  /* #ifndef  TWODIMS */
  hinv3 = hinv * hinv / boxSize_Z;
#endif /* #ifndef  TWODIMS #else */
  hinv4 = hinv3 * hinv;
 
/*bh timestep*/
      dt    = (bin ? (((integertime)1) << bin) : 0) * All.Timebase_interval;
    //dtime = All.cf_atime * dt / All.cf_time_hubble_a;

  if(isbh)/*is bh->*/
    {
#ifdef BONDI_ACCRETION
      energyfeed = All.Epsilon_f * All.Epsilon_r * accretion_rate * dt * (CLIGHT * CLIGHT / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s));
#endif
#ifdef INFALL_ACCRETION  
      energyfeed = All.Epsilon_f * All.Epsilon_r * accretion * (CLIGHT * CLIGHT / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s));
#endif
    }
  else if(!isbh)/*is star->*/
    {
      double EddingtonLuminosity = 4. * M_PI * GRAVITY * (bh_mass * All.UnitMass_in_g) * PROTONMASS * CLIGHT / THOMPSON;
      EddingtonLuminosity *=  (All.UnitTime_in_s / (All.UnitMass_in_g*pow(All.UnitVelocity_in_cm_per_s,2)));
      energyfeed = EddingtonLuminosity * dt;
    }

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

      dx = pos[0] - P[j].Pos[0];
      dy = pos[1] - P[j].Pos[1];
      dz = pos[2] - P[j].Pos[2];

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

/*kernel*/
      if(r2 < h2)
        {
          r = sqrt(r2);

          u = r * hinv;

          kernel(u, hinv3, hinv4, &wk, &dwk);

          if(isbh)/*particle is a bh*/
            {
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
/*set flag to 1 if gas particle is on the positive side of jet*/
                  if(pos_x_angle <= theta)
                    SphP[j].PositiveJet = 1;
/*check if particle is inside the cone*/ 
                  if((pos_x_angle <= theta) || (neg_x_angle <= theta))
                    {
/*split kinetic and thermal energy feed*/ 
                      SphP[j].KineticFeed       += (1-All.Ftherm) * energyfeed/ngbmass_feed*P[j].Mass;
                      All.EnergyExchange[0]     += (1-All.Ftherm) * energyfeed/ngbmass_feed*P[j].Mass;

/*only jet particles injected with thermal feedback if JetFeedback == 2, else isotropic*/             
                      if(All.JetFeedback == 2)     
                        {
                          SphP[j].ThermalFeed   += All.Ftherm * energyfeed/ngbmass_feed*P[j].Mass;
                          All.EnergyExchange[0] += All.Ftherm * energyfeed/ngbmass_feed*P[j].Mass;
                        }
                    }
              
                  if(All.JetFeedback == 1)
                    {
                      SphP[j].ThermalFeed   += All.Ftherm * energyfeed/ngbmass*P[j].Mass;
                      All.EnergyExchange[0] += All.Ftherm * energyfeed/ngbmass*P[j].Mass;
                    } 
            
                }
/*else All.JetFeedback == 0 i.e. isotropic thermal + kinetic injection*/
              else
                {
                  SphP[j].KineticFeed   += (1-All.Ftherm) * energyfeed/ngbmass*P[j].Mass;
                  All.EnergyExchange[0] += (1-All.Ftherm) * energyfeed/ngbmass*P[j].Mass;
                  SphP[j].ThermalFeed   += All.Ftherm * energyfeed/ngbmass*P[j].Mass;
                  All.EnergyExchange[0] += All.Ftherm * energyfeed/ngbmass*P[j].Mass;
                }
#ifdef BONDI_ACCRETION
/*set drain mass flag*/
              SphP[j].MassDrain = accretion_rate*dt/ngbmass*P[j].Mass + mass_to_drain/ngbmass*P[j].Mass;
#endif
/*set radial kick direction*/      
              SphP[j].BhKickVector[0] = -dx;
              SphP[j].BhKickVector[1] = -dy;
              SphP[j].BhKickVector[2] = -dz;
            }
          
          if(!isbh) /*particle is a star*/
            {
/*set radial momentum kick*/
              SphP[j].MomentumFeed  += All.Lambda * energyfeed / (CLIGHT / All.UnitVelocity_in_cm_per_s) * P[j].Mass / bh_rho * wk;
/*Epsilon_r terms cancel out in the eddington rate formula*/

              All.EnergyExchange[2] += All.Lambda * energyfeed / (CLIGHT / All.UnitVelocity_in_cm_per_s) * P[j].Mass / bh_rho * wk;

              SphP[j].MomentumKickVector[0] = -dx;
              SphP[j].MomentumKickVector[1] = -dy;
              SphP[j].MomentumKickVector[2] = -dz;
            }
        }
    }
  return 0;
}
  /* Now collect the result at the right place 
  if(mode == MODE_LOCAL_PARTICLES)
    out2particle(&out, target, MODE_LOCAL_PARTICLES);
  else
    DataResult[target] = out;

  return 0;
}*/

