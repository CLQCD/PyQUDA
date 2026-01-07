#include <malloc_quda.h>

extern "C" {
void *cupy_malloc(void *param, size_t size, int device) { return pool_device_malloc(size); }
void cupy_free(void *param, void *ptr, int device) { pool_device_free(ptr); }
void *torch_malloc(size_t size, int device, void *stream) { return pool_device_malloc(size); }
void torch_free(void *ptr, int device, void *stream) { pool_device_free(ptr); }
}
