// NE PAS MODIFIER
static int4 color_to_int4(unsigned c) {
    uchar4 ci = *(uchar4 * ) & c;
    return convert_int4(ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color(int4 i) {
    return (unsigned) convert_uchar4(i);
}


// NE PAS MODIFIER
static float4 color_scatter(unsigned c) {
    uchar4 ci;

    ci.s0123 = (*((uchar4 * ) & c)).s3210;
    return convert_float4(ci) / (float4) 255;
}

static unsigned is_alive(__global unsigned *in, unsigned i, unsigned j) {
    return in[i * DIM + j] != 0;
}

static unsigned count_neighbors(__global unsigned *in, unsigned i, unsigned j) {
    int start_x = (i == 0) ? i : i - 1;
    int start_y = (j == 0) ? j : j - 1;
    int end_x = (i == DIM - 1) ? i : i + 1;
    int end_y = (j == DIM - 1) ? j : j + 1;

    unsigned sum = 0;

    for (int x = start_x; x <= end_x; x++) {
        for (int y = start_y; y <= end_y; y++) {
            if (x != i || y != j) {
                sum += is_alive(in, x, y);
            }
        }
    }

    return sum;
}

static unsigned change_color(__global unsigned* in, int i, int j) {
    int nb_neighbor = count_neighbors(in, i, j);
    if (nb_neighbor < 2 || nb_neighbor > 3)
        return 0;
    else if (nb_neighbor == 2 && in[i * DIM + j] != 0)
        return in[i * DIM + j];
    else if (nb_neighbor == 3)
        return int4_to_color((int4)(255, 0, 255, 255));
    else
        return 0;
}

static unsigned is_alive_local(__local unsigned in[TILEX + 2][TILEY + 2], unsigned i, unsigned j) {
    return in[i][j] != 0;
}

static unsigned count_neighbors_local(__local unsigned in[TILEX + 2][TILEY + 2], unsigned i, unsigned j) {
    unsigned start_x = i - 1;
    unsigned start_y = j - 1;
    unsigned end_x = i + 1;
    unsigned end_y = j + 1;

    unsigned sum = 0;

    for (unsigned x = start_x; x <= end_x; x++) {
        for (unsigned y = start_y; y <= end_y; y++) {
            if (x != i || y != j) {
                sum += is_alive_local(in, x, y);
            }
        }
    }

    return sum;
}

static unsigned change_color_local(__local unsigned in[TILEX + 2][TILEY + 2], unsigned i, unsigned j) {
    int nb_neighbor = count_neighbors_local(in, i, j);
    if (nb_neighbor < 2 || nb_neighbor > 3)
        return 0;
    else if (nb_neighbor == 2 && in[i][j] != 0)
        return in[i][j];
    else if (nb_neighbor == 3)
        return int4_to_color((int4)(255, 0, 255, 255));
    else
        return 0;
}

__kernel void transpose_naif(__global unsigned *in, __global unsigned *out) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    out[x * DIM + y] = in[y * DIM + x];
}


__kernel void transpose(__global unsigned *in, __global unsigned *out) {
    __local unsigned tile[TILEX][TILEY + 1];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int xloc = get_local_id(0);
    int yloc = get_local_id(1);

    tile[xloc][yloc] = in[y * DIM + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    out[(x - xloc + yloc) * DIM + y - yloc + xloc] = tile[yloc][xloc];
}

__kernel void compute_naif(__global unsigned *in, __global unsigned *out) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    out[x * DIM + y] = change_color(in, x, y);
}

__kernel void compute(__global unsigned *in, __global unsigned *out, __global unsigned *tile_in, __global unsigned *tile_out) {
    __local unsigned tile[TILEX + 2][TILEY + 2];

    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    unsigned xloc = get_local_id(0);
    unsigned yloc = get_local_id(1);
    unsigned xg = get_group_id(0);
    unsigned yg = get_group_id(1);

        tile[xloc + 1][yloc + 1] = in[x * DIM + y];

        if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
            if (xloc == 0) {
                tile[0][yloc + 1] = in[(x - 1) * DIM + y];
            } else if (xloc == TILEX - 1) {
                tile[TILEX + 1][yloc + 1] = in[(x + 1) * DIM + y];
            }

            if (yloc == 0) {
                tile[xloc + 1][0] = in[x * DIM + (y - 1)];
            } else if (yloc == TILEY - 1) {
                tile[xloc + 1][TILEY + 1] = in[x * DIM + (y + 1)];
            }

            if (xloc == 0) {
                if (yloc == 0) {
                    tile[0][0] = in[(x - 1) * DIM + (y - 1)];
                } else if (yloc == TILEY - 1){
                    tile[0][TILEY + 1] = in[(x - 1) * DIM + (y + 1)];
                }
            } else if (xloc == TILEX - 1) {
                if (yloc == 0) {
                    tile[TILEX + 1][0] = in[(x + 1) * DIM + (y - 1)];
                } else if (yloc == TILEY - 1){
                    tile[TILEX + 1][TILEY + 1] = in[(x + 1) * DIM + (y + 1)];
                }
            }
        } else {
            if (xloc == 0) {
                tile[0][yloc + 1] = 0;
            } else if (xloc == TILEX - 1) {
                tile[TILEX + 1][yloc + 1] = 0;
            }

            if (yloc == 0) {
                tile[xloc + 1][0] = 0;
            } else if (yloc == TILEY - 1) {
                tile[xloc + 1][TILEY + 1] = 0;
            }

            if (xloc == 0) {
                if (yloc == 0) {
                    tile[0][0] = 0;
                } else if (yloc == TILEY - 1){
                    tile[0][TILEY + 1] = 0;
                }
            } else if (xloc == TILEX - 1) {
                if (yloc == 0) {
                    tile[TILEX + 1][0] = 0;
                } else if (yloc == TILEY - 1){
                    tile[TILEX + 1][TILEY + 1] = 0;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned new_value = change_color_local(tile, xloc + 1, yloc + 1);
        tile[xloc + 1][yloc + 1] = new_value;

        out[x * DIM + y] = tile[xloc + 1][yloc + 1];
}


// NE PAS MODIFIER
static unsigned color_mean(unsigned c1, unsigned c2) {
    uchar4 c;

    c.x = ((unsigned) (((uchar4 * ) & c1)->x) + (unsigned) (((uchar4 * ) & c2)->x)) / 2;
    c.y = ((unsigned) (((uchar4 * ) & c1)->y) + (unsigned) (((uchar4 * ) & c2)->y)) / 2;
    c.z = ((unsigned) (((uchar4 * ) & c1)->z) + (unsigned) (((uchar4 * ) & c2)->z)) / 2;
    c.w = ((unsigned) (((uchar4 * ) & c1)->w) + (unsigned) (((uchar4 * ) & c2)->w)) / 2;

    return (unsigned) c;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture(__global unsigned *cur, __write_only image2d_t tex) {
    int y = get_global_id(1);
    int x = get_global_id(0);
    int2 pos = (int2)(x, y);
    unsigned c;

    c = cur[y * DIM + x];

    write_imagef(tex, pos, color_scatter(c));
}
