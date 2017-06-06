
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>
#include <omp.h>

unsigned version = 0;

unsigned compute_v0(unsigned nb_iter);

unsigned compute_v1(unsigned nb_iter);

unsigned compute_v2(unsigned nb_iter);

unsigned compute_v3(unsigned nb_iter);

unsigned compute_v4(unsigned nb_iter);

unsigned compute_v5(unsigned nb_iter);

unsigned compute_v6(unsigned nb_iter);

unsigned compute_v7(unsigned nb_iter);

unsigned compute_v8(unsigned nb_iter);

unsigned compute_v9(unsigned nb_iter);

void_func_t first_touch[] = {
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
};

int_func_t compute[] = {
        compute_v0,
        compute_v1,
        compute_v2,
        compute_v3,
        compute_v4,
        compute_v5,
        compute_v6,
        compute_v7,
        compute_v8,
        compute_v9,
};

char *version_name[] = {
        "Séquentielle",
        "Séquentielle Tuilée",
        "Séquentielle Tuilée Optimisée",
        "OpenMP",
        "OpenMP Tuilée",
        "OpenMP Tuilée Optimisée",
        "OpenMP Task Tuilée",
        "OpenMP Task Tuilée Optimisée",
        "OpenCL",
        "OpenCL Optimisée",
};

unsigned opencl_used[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
};

unsigned opencl_tiles_used[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
};

const char *opencl_kernel_names[] = {
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        "compute_naif",
        "compute"
};

int is_alive(int i, int j) {
    return cur_img(i, j) != 0;
}

bool* tile_curr;
bool* tile_next;
int* tile_state;


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

bool change_color(int i, int j) {
    int state_before = cur_img(i, j);
    int nb_neighbor = count_neighbors(i, j);
    if (nb_neighbor < 2 || nb_neighbor > 3)
        next_img(i, j) = 0;
    else if (nb_neighbor == 2 && cur_img(i, j) != 0)
        next_img(i, j) = cur_img(i, j);
    else if (nb_neighbor == 3)
        next_img(i, j) = get_color(255, 255, 0);
    else
        next_img(i, j) = 0;

    return (state_before != next_img(i, j));

}

void init_changed_tiles() {
    int nb_tiles = (int) ceil(DIM * 1.0f / TILE);

    tile_curr = malloc(nb_tiles * nb_tiles * sizeof(bool));
    tile_next = malloc(nb_tiles * nb_tiles * sizeof(bool));

    for (int i = 0; i < nb_tiles * nb_tiles; ++i) {
        tile_curr[i] = true;
        tile_next[i] = true;
    }
}

///////////////////////////// Version séquentielle simple


unsigned compute_v0(unsigned nb_iter) {

    for (unsigned it = 1; it <= nb_iter; it++) {
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) {
                change_color(i, j);
            }
        }
        swap_images();
    }
    // retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return 0;
}

///////////////////////////// Version séquentielle tuilée

unsigned compute_v1(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; ++it) {
        for (unsigned i = 0; i < DIM; i += TILE) {
            for (unsigned j = 0; j < DIM; j += TILE) {

                unsigned end_i = i + TILE < DIM ? i + TILE : DIM;
                unsigned end_j = j + TILE < DIM ? j + TILE : DIM;


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
    int nb_tiles = (int) ceil(DIM * 1.0f / TILE);

    for (unsigned it = 1; it <= nb_iter; ++it) {
        for (unsigned i = 0; i < DIM; i += TILE) {
            for (unsigned j = 0; j < DIM; j += TILE) {
                unsigned x_tile = i / TILE;
                unsigned y_tile = j / TILE;

                // Si la tuile a changé à l'itération précédente alors on la recalcule
                if (tile_curr[x_tile * nb_tiles + y_tile]) {
                    tile_next[x_tile * nb_tiles + y_tile] = false;

                    unsigned end_i = i + TILE < DIM ? i + TILE : DIM;
                    unsigned end_j = j + TILE < DIM ? j + TILE : DIM;

                    for (unsigned i_tile = i; i_tile < end_i; ++i_tile) {
                        for (unsigned j_tile = j; j_tile < end_j; ++j_tile) {
                            if (change_color(i_tile, j_tile)) {
                                tile_next[x_tile * nb_tiles + y_tile] = true;

                                if (i_tile == end_i - 1 && x_tile != nb_tiles - 1) {    //bas
                                    tile_next[(x_tile + 1) * nb_tiles + y_tile] = true;
                                } else if (i_tile == i && x_tile != 0) {                //haut
                                    tile_next[(x_tile - 1) * nb_tiles + y_tile] = true;
                                }

                                if (j_tile == end_j - 1 && y_tile != nb_tiles - 1) {    //droite
                                    tile_next[x_tile * nb_tiles + (y_tile + 1)] = true;
                                } else if (j_tile == j && y_tile != 0) {                //gauche
                                    tile_next[x_tile * nb_tiles + (y_tile - 1)] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        swap_images();
        bool* tmp = tile_curr;
        tile_curr = tile_next;
        tile_next = tmp;
    }
    return 0;
}

///////////////////////////// Version OpenMP for de base

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

unsigned compute_v3(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) {
                change_color(i, j);
            }
        }
        swap_images();
    }
    return 0;
}

///////////////////////////// Version OpenMP for tuilée

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

unsigned compute_v4(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; ++it) {
        #pragma omp parallel for collapse(2) schedule(static)
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
    int nb_tiles = (int) ceil(DIM * 1.0f / TILE);

    for (unsigned it = 1; it <= nb_iter; ++it) {

        #pragma omp parallel for collapse(2) schedule(static)
        for (unsigned i = 0; i < DIM; i += TILE) {
            for (unsigned j = 0; j < DIM; j += TILE) {
                unsigned x_tile = i / TILE;
                unsigned y_tile = j / TILE;

                // Si la tuile a changé à l'itération précédente alors on la recalcule
                if (tile_curr[x_tile * nb_tiles + y_tile]) {
                    tile_next[x_tile * nb_tiles + y_tile] = false;

                    unsigned end_i = i + TILE < DIM ? i + TILE : DIM;
                    unsigned end_j = j + TILE < DIM ? j + TILE : DIM;

                    for (unsigned i_tile = i; i_tile < end_i; ++i_tile) {
                        for (unsigned j_tile = j; j_tile < end_j; ++j_tile) {
                            if (change_color(i_tile, j_tile)) {
                                tile_next[x_tile * nb_tiles + y_tile] = true;

                                if (i_tile == end_i - 1 && x_tile != nb_tiles - 1) {    //bas
                                    tile_next[(x_tile + 1) * nb_tiles + y_tile] = true;
                                } else if (i_tile == i && x_tile != 0) {                //haut
                                    tile_next[(x_tile - 1) * nb_tiles + y_tile] = true;
                                }

                                if (j_tile == end_j - 1 && y_tile != nb_tiles - 1) {    //droite
                                    tile_next[x_tile * nb_tiles + (y_tile + 1)] = true;
                                } else if (j_tile == j && y_tile != 0) {                //gauche
                                    tile_next[x_tile * nb_tiles + (y_tile - 1)] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        swap_images();
        bool* tmp = tile_curr;
        tile_curr = tile_next;
        tile_next = tmp;
    }
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
    int nb_tiles = (int) ceil(DIM * 1.0f / TILE);

    for (unsigned it = 1; it <= nb_iter; ++it) {
        #pragma omp parallel
        for (unsigned i = 0; i < DIM; i += TILE)
            for (unsigned j = 0; j < DIM; j += TILE) {
                unsigned x_tile = i / TILE;
                unsigned y_tile = j / TILE;
                if (tile_curr[x_tile * nb_tiles + y_tile]){
                    #pragma omp single nowait
                    #pragma omp task
                    {
                        tile_next[x_tile * nb_tiles + y_tile] = false;
                        unsigned end_i = i + TILE < DIM ? i + TILE : DIM;
                        unsigned end_j = j + TILE < DIM ? j + TILE : DIM;

                        for (unsigned i_tile = i; i_tile < end_i; ++i_tile) {
                            for (unsigned j_tile = j; j_tile < end_j; ++j_tile) {
                                if (change_color(i_tile, j_tile)) {
                                    tile_next[x_tile * nb_tiles + y_tile] = true;

                                    if (i_tile == end_i - 1 && x_tile != nb_tiles - 1) {    //bas
                                        tile_next[(x_tile + 1) * nb_tiles + y_tile] = true;
                                    } else if (i_tile == i && x_tile != 0) {                //haut
                                        tile_next[(x_tile - 1) * nb_tiles + y_tile] = true;
                                    }

                                    if (j_tile == end_j - 1 && y_tile != nb_tiles - 1) {    //droite
                                        tile_next[x_tile * nb_tiles + (y_tile + 1)] = true;
                                    } else if (j_tile == j && y_tile != 0) {                //gauche
                                        tile_next[x_tile * nb_tiles + (y_tile - 1)] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        swap_images ();
        bool* tmp = tile_curr;
        tile_curr = tile_next;
        tile_next = tmp;
    }

//
//
//
//
//    for (unsigned it = 1; it <= nb_iter; ++it) {
//        #pragma omp parallel
//        for (unsigned i = 0; i < DIM; i += TILE) {
//            for (unsigned j = 0; j < DIM; j += TILE) {
//                unsigned x_tile = i / TILE;
//                unsigned y_tile = j / TILE;
//
//                // Si la tuile a changé à l'itération précédente alors on la recalcule
//                if (tile_curr[x_tile * nb_tiles + y_tile]) {
//                    tile_next[x_tile * nb_tiles + y_tile] = false;
//
//                    unsigned end_i = i + TILE < DIM ? i + TILE : DIM;
//                    unsigned end_j = j + TILE < DIM ? j + TILE : DIM;
//
//                    for (unsigned i_tile = i; i_tile < end_i; ++i_tile) {
//                        for (unsigned j_tile = j; j_tile < end_j; ++j_tile) {
//                            if (change_color(i_tile, j_tile)) {
//                                #pragma omp single nowait
//                                #pragma omp task
//                                {
//                                    tile_next[x_tile * nb_tiles + y_tile] = true;
//
//                                    if (i_tile == i && x_tile != 0) {                                // a gauche
//                                        if (y_tile > 0) {
//                                            tile_next[(x_tile - 1) * nb_tiles + (y_tile - 1)] = true;
//                                        }
//                                        tile_next[(x_tile - 1) * nb_tiles + y_tile] = true;
//                                        if (y_tile < nb_tiles - 1) {
//                                            tile_next[(x_tile - 1) * nb_tiles + (y_tile + 1)] = true;
//                                        }
//                                    } else if (i_tile == end_i - 1 && x_tile != nb_tiles - 1) {    // a droite
//                                        if (y_tile > 0) {
//                                            tile_next[(x_tile + 1) * nb_tiles + (y_tile - 1)] = true;
//                                        }
//                                        tile_next[(x_tile + 1) * nb_tiles + y_tile] = true;
//                                        if (y_tile < nb_tiles - 1) {
//                                            tile_next[(x_tile + 1) * nb_tiles + (y_tile + 1)] = true;
//                                        }
//                                    }
//
//                                    if (j_tile == j && y_tile != 0) {                                // en haut
//                                        if (x_tile > 0) {
//                                            tile_next[(x_tile - 1) * nb_tiles + (y_tile - 1)] = true;
//                                        }
//                                        tile_next[x_tile * nb_tiles + (y_tile - 1)] = true;
//                                        if (x_tile < nb_tiles - 1) {
//                                            tile_next[(x_tile + 1) * nb_tiles + (y_tile - 1)] = true;
//                                        }
//                                    } else if (j_tile == end_j - 1 && y_tile != nb_tiles - 1) {    // en bas (ces soirées là)
//                                        if (x_tile > 0) {
//                                            tile_next[(x_tile - 1) * nb_tiles + (y_tile + 1)] = true;
//                                        }
//                                        tile_next[x_tile * nb_tiles + (y_tile + 1)] = true;
//                                        if (x_tile < nb_tiles - 1) {
//                                            tile_next[(x_tile + 1) * nb_tiles + (y_tile + 1)] = true;
//                                        }
//                                    }
//                                }
//                            }
//                        }
//
//                    }
//                }
//            }
//        }
//        swap_images();
//        bool* tmp = tile_curr;
//        tile_curr = tile_next;
//        tile_next = tmp;
//    }

    return 0;
}

///////////////////////////// Version OpenCL de base

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v8(unsigned nb_iter) {
    return ocl_compute(nb_iter);
}

///////////////////////////// Version OpenCL optimisée

unsigned compute_v9(unsigned nb_iter) {
    return ocl_compute_with_tiles(nb_iter);
}