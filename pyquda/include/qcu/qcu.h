#pragma once

typedef struct Shape {
  int dims_[6];
} Shape;

#ifdef __cplusplus
extern "C" {
#endif

void my_dslash_interface(void *U_ptr, void *a_ptr, void *b_ptr, int Lx, int Ly, int Lz, int Lt, int Nd, int Ns, int Nc);

#ifdef __cplusplus
}
#endif
