/*black hole functions*/

void reallocate_memory_maxpartbh(void);
void domain_resize_storage_bh(int count_get_bh);

void kernel(double u, double hinv3, double hinv4, double *wk, double *dwk);
void bh_density(void);
void update_bh_accretion_rate(void);
void bh_ngb_feedback(void);

integertime get_timestep_bh(int p);
void update_bh_timesteps(void);
void reconstruct_bh_timebins(void);
void update_list_of_active_bh_particles(void);
/*void update_list_of_active_bh_particles_prior_mesh(void);*/

void perform_end_of_step_bh_physics(void);

