
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

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


///////////////////////////// Version séquentielle simple

int count_neighbors(int i, int j) {
  int start_x = (i == 0) ? i : i - 1;
  int start_y = (j == 0) ? j : j - 1;
  int end_x = (i == DIM - 1) ? i : i + 1;
  int end_y = (j == DIM - 1) ? j : j + 1;

  int sum = 0;

  for (int x = start_x; x <= end_x ; x++) {
    for (int y = start_y; y <= end_y ; y++) {
      if (x != i || y != j) {
        sum += is_alive(x, y);
      }
    }
  }

  return sum;
}

unsigned compute_v0 (unsigned nb_iter) {

  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        int nb_neighbor = count_neighbors(i, j);
        if (nb_neighbor < 2 || nb_neighbor > 3)
          next_img (i, j) = 0;
        else if (nb_neighbor == 2 && cur_img(i, j) != 0)
          next_img(i, j) = cur_img(i, j);
        else if (nb_neighbor == 3)
          next_img(i, j) = get_color(255,0,255);
        else
          next_img (i, j) = 0;
     }
    }
    swap_images ();
  }
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}


///////////////////////////// Version OpenMP de base

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
unsigned compute_v1(unsigned nb_iter) {
  #pragma omp parallel for
  for (unsigned it = 1; it <= nb_iter; it ++) {
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        int nb_neighbor = count_neighbors(i, j);
        if (nb_neighbor < 2 || nb_neighbor > 3)
          next_img (i, j) = 0;
        else if (nb_neighbor == 2 && cur_img(i, j) != 0)
          next_img(i, j) = cur_img(i, j);
        else if (nb_neighbor == 3)
          next_img(i, j) = get_color(255,0,255);
        else
          next_img (i, j) = 0;
     }
    }
    swap_images ();
  }
  return 0;
}



///////////////////////////// Version OpenMP optimisée

void first_touch_v2 ()
{

}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v2(unsigned nb_iter)
{
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter)
{
  return ocl_compute (nb_iter);
}
