
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

unsigned TILE = 32;
unsigned version = 0;

void first_touch_v1 (void);
void first_touch_v2 (void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);

void_func_t first_touch [] = {
  NULL,
  first_touch_v1,
  first_touch_v2,
  NULL,
};

int_func_t compute [] = {
  compute_v0,
  compute_v1,
  compute_v2,
  compute_v3,
};

char *version_name [] = {
  "Séquentielle",
  "OpenMP",
  "OpenMP zone",
  "OpenCL",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  1,
};

int is_alive(int i, int j) {
  return cur_img(i, j) != 0;
}


int count_neighbors(int i, int j) {
	int start_x = (i == 0) ? i : i - 1;
	int start_y = (j == 0) ? j : j - 1;
	int end_x = (i == DIM - 1) ? i : i + 1;
	int end_y = (j == DIM - 1) ? j : j + 1;

	int sum = 0;

	for (int x = start_x; x <= end_x; x++) {
		for (int y = start_y; y <= end_y; y++) {
			if (x != i || y != j) {
				sum += is_alive(x, y);
			}
		}
	}

	return sum;
}

void change_color(int i, int j) {
	int nb_neighbor = count_neighbors(i, j);
	if (nb_neighbor < 2 || nb_neighbor > 3)
		next_img(i, j) = 0;
	else if (nb_neighbor == 2 && cur_img(i, j) != 0)
		next_img(i, j) = cur_img(i, j);
	else if (nb_neighbor == 3)
		next_img(i, j) = get_color(255, 0, 255);
	else
		next_img(i, j) = 0;

}

///////////////////////////// Version séquentielle simple


unsigned compute_v0 (unsigned nb_iter) {

  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
		  change_color(i, j);
     }
    }
    swap_images ();
  }
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

///////////////////////////// Version séquentielle tuilée

unsigned compute_v1 (unsigned nb_iter) {
	for (unsigned it = 1; it <= nb_iter; ++it) {
		for (unsigned i = 0; i < DIM - 1; i += TILE) {
			for (unsigned j = 0; j < DIM - 1; j += TILE) {
				unsigned end_i = i + TILE < DIM - 1 ? i + TILE : DIM - 1;
				unsigned end_j = j + TILE < DIM - 1 ? j + TILE : DIM - 1;
					for (unsigned i_tile = i; i_tile < end_i; ++i_tile) {
						for (unsigned j_tile = j; j_tile < end_j; ++j_tile) {
						change_color(i_tile, j_tile);
					}
				}
			}
		}
		swap_images();
	}
	return 0;
}
///////////////////////////// Version séquentielle optimisée

unsigned compute_v2(unsigned nb_iter) {

	return 0;
}

///////////////////////////// Version OpenMP for de base

void first_touch_v1 ()
{
  int i,j ;

 #pragma omp parallel for
  for(i=0; i<DIM ; i++) {
    for(j=0; j < DIM ; j += 512) {
      next_img (i, j) = cur_img (i, j) = 0 ;
    }
  }
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

unsigned compute_v3(unsigned nb_iter) {
  #pragma omp parallel for
  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
		  change_color(i, j);
     }
    }
    swap_images ();
  }
  return 0;
}

///////////////////////////// Version OpenMP for tuilée

void first_touch_v2 ()
{

}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

unsigned compute_v4(unsigned nb_iter)
{
	for (unsigned it = 1; it <= nb_iter; ++it) {
		#pragma omp parallel for //TODO : collapse(2) schedule(static) ??
		for (unsigned i = 0; i < DIM - 1; i += TILE) {
			for (unsigned j = 0; j < DIM - 1; j += TILE) {
				unsigned end_i = i + TILE < DIM - 1 ? i + TILE : DIM - 1;
				unsigned end_j = j + TILE < DIM - 1 ? j + TILE : DIM - 1;
				for (unsigned i_tile = i; i_tile < end_i; ++i_tile) {
					for (unsigned j_tile = j; j_tile < end_j; ++j_tile) {
						change_color(i_tile, j_tile);
					}
				}
			}
		}
		swap_images();
	}

  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenMP for optimisée

unsigned compute_v5(unsigned nb_iter) {

	return 0;
}


///////////////////////////// Version OpenMP task tuilée

unsigned compute_v6(unsigned nb_iter) {

	for (unsigned it = 1; it <= nb_iter; ++it) {
		#pragma omp parallel
		for (unsigned i = 0; i < DIM - 1; i += TILE)
			for (unsigned j = 0; j < DIM - 1; j += TILE) {
			#pragma omp single nowait
			#pragma omp task //TODO shared ou local les variables ?
				{
					unsigned end_i = i + TILE < DIM - 1 ? i + TILE : DIM - 1;
					unsigned end_j = j + TILE < DIM - 1 ? j + TILE : DIM - 1;
					for (unsigned i_tile = i; i_tile < end_i; ++i_tile)
						for (unsigned j_tile = j; j_tile < end_j; ++j_tile)
							change_color(i_tile, j_tile);
				}
			}
		swap_images();
	}
	return 0;
}

///////////////////////////// Version OpenMP task optimisée

unsigned compute_v7(unsigned nb_iter) {

	return 0;
}

///////////////////////////// Version OpenCL de base

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v8 (unsigned nb_iter)
{
  return ocl_compute (nb_iter);
}

///////////////////////////// Version OpenCL optimisée

unsigned compute_v9(unsigned nb_iter) {

	return 0;
}