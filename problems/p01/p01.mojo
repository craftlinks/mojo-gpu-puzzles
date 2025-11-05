from memory import UnsafePointer
from gpu import thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add_10
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE
alias dtype = DType.float32


# Kernel function to add 10 to each element of an array. ('fn')
fn add_10(
    output: UnsafePointer[Scalar[dtype]], a: UnsafePointer[Scalar[dtype]]
):
    i = thread_idx.x
    output[i] = a[i] + 10


# ANCHOR_END: add_10


def main():
    # Initialize GPU context for device 0 (default GPU device).
    with DeviceContext() as ctx:
        # Create a buffer in device (GPU) memory to store data for computation.
        # Fill the buffer with zeros.
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out = out.enqueue_fill(0)

        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a = a.enqueue_fill(0)

        # Maps this device buffer to host memory for CPU access.
        # Values modified inside the with statement are updated on the device
        # when the with statement exits.
        # https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceBuffer/#map_to_host

        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i

        # # Compile the kernel function for execution on the GPU.
        compiled_add_10 = ctx.compile_function_checked[add_10, add_10]()

        # Launch the GPU kernel
        ctx.enqueue_function_checked(
            compiled_add_10,
            out,
            a,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )


        # Create a buffer in host (CPU) memory to store expected results.
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected = expected.enqueue_fill(0)

        # Blocks until all asynchronous calls on the stream associated with this device context have completed.
        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = i + 10

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
