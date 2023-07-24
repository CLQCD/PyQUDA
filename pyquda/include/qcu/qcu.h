#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
  int lattice_size[4];
} QcuParam;

void dslashQcu(void *fermion_in, void *fermion_out, void *gauge, QcuParam *param);

#ifdef __cplusplus
}
#endif
