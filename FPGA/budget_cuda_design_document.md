# Budget-CUDA Zynq INT8 GEMM Accelerator Design Document

## 0. Project Summary

Budget-CUDA is a lightweight FPGA-backed inference runtime for a Zynq platform. The system uses the Zynq PL as a fixed-function INT8 GEMM accelerator and uses the Zynq PS as the local device runtime. The host computer provides a Python API, model conversion tools, quantization utilities, and a protocol client.

The project goal is not to implement a full CUDA replacement. The goal is to build a small accelerator stack that behaves like a simple device runtime:

```text
Host Python API
    -> Budget-CUDA protocol
        -> Zynq PS runtime
            -> Zynq PL INT8 GEMM accelerator
                -> Zynq DRAM resident tensors
```

The accelerator supports INT8 inference. Floating-point execution is not supported by the PL. Models must be quantized to INT8 before deployment. GEMM operations run on the PL. Non-GEMM operations run on the PS unless later fused into the PL.

---

## 1. Top-Level System Architecture

```text
+--------------------------------------------------------------+
|                        Host Computer                         |
|                                                              |
|  Python API / Model Converter / Protocol Client              |
|                                                              |
+-----------------------------+--------------------------------+
                              |
                              | ALLOC / WRITE / READ / EXECUTE / FREE
                              v
+--------------------------------------------------------------+
|                         Zynq PS                              |
|                                                              |
|  Protocol Server                                             |
|  Device Memory Manager                                       |
|  Model/Graph Runtime                                         |
|  Non-GEMM Operator Runtime                                   |
|  PL Driver                                                   |
|                                                              |
+-----------------------------+--------------------------------+
                              |
                              | AXI-Lite Control + AXI/DMA Data
                              v
+--------------------------------------------------------------+
|                         Zynq PL                              |
|                                                              |
|  GEMM Controller                                             |
|  Tile Load/Store Engine                                      |
|  BRAM/SRAM Tile Caches                                       |
|  8 x 16x16 INT8 Systolic Arrays                              |
|  INT32 Accumulators                                          |
|  Optional Requantization / Activation                        |
|                                                              |
+--------------------------------------------------------------+
```

### 1.1 System Ownership Split

| Task | Host Computer | Zynq PS | Zynq PL |
|---|---:|---:|---:|
| Python user API | Yes | No | No |
| Model import/conversion | Yes | No | No |
| INT8 quantization | Yes | Maybe | No |
| Model lowering | Yes | No | No |
| Weight upload | Yes | Yes | No |
| Device-memory allocation | Requests only | Owns allocation | No |
| Physical address management | No | Yes | No |
| Protocol parsing | No | Yes | No |
| Model graph scheduling | No | Yes | No |
| Non-GEMM operators | No | Yes, initially | No |
| GEMM execution | No | Starts/monitors | Yes |
| INT8 systolic compute | No | No | Yes |
| DMA/cache handling | No | Yes | Yes, data path |
| Output download | Yes | Yes | No |

---

## 2. Design Principles

### 2.1 Keep the External Protocol Small

The external host-to-PS protocol has only five commands:

```text
ALLOC
WRITE
READ
EXECUTE
FREE
```

Do not add separate transport commands for GEMM, YOLO, model loading, ReLU, convolution, or graph execution. Those are represented as operation types or descriptors inside `EXECUTE`.

### 2.2 Use Device Memory Handles

The host never receives physical addresses. The host receives logical buffer handles.

```text
Host-visible:
    buffer_id = 12

PS-internal:
    buffer_id 12 -> physical DDR address 0x18040000
```

The PS translates buffer IDs into physical addresses before programming the PL.

### 2.3 Keep Model Data Resident on the Device

Model weights should be uploaded once and stored in Zynq DRAM. During inference, the host should send only the input tensor and read only the final output.

Bad pattern:

```text
For every layer:
    host sends weights
    host sends activation
    host reads activation
```

Correct pattern:

```text
Initialization:
    host uploads all weights and graph descriptors once

Inference:
    host writes input tensor
    host executes graph
    host reads final output only
```

### 2.4 Keep the PL Simple

The PL should not understand Python, CUDA, YOLO, CNNs, ONNX, or graph descriptors. It should only understand physical addresses, matrix dimensions, flags, and start/done control.

The PL receives:

```text
A physical address
B physical address
C physical address
M
N
K
flags
start
```

The PL produces:

```text
C result in Zynq DRAM
done/status flag
```

---

## 3. Repository Structure

Recommended repository layout:

```text
budget-cuda/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── protocol.md
│   ├── memory_model.md
│   ├── graph_format.md
│   ├── hardware_design.md
│   ├── quantization.md
│   └── benchmarks.md
│
├── common/
│   ├── bcuda_protocol.h
│   ├── bcuda_protocol.py
│   ├── bcuda_graph.h
│   ├── bcuda_graph.py
│   ├── bcuda_dtypes.h
│   └── bcuda_dtypes.py
│
├── host/
│   ├── budgetcuda/
│   │   ├── __init__.py
│   │   ├── device.py
│   │   ├── tensor.py
│   │   ├── model.py
│   │   ├── graph.py
│   │   ├── ops.py
│   │   ├── protocol.py
│   │   ├── serializer.py
│   │   ├── quantization.py
│   │   └── converter/
│   │       ├── __init__.py
│   │       ├── torch_import.py
│   │       ├── onnx_import.py
│   │       ├── yolo_import.py
│   │       └── memory_plan.py
│   │
│   ├── examples/
│   │   ├── gemm_demo.py
│   │   ├── mlp_demo.py
│   │   ├── cnn_demo.py
│   │   └── yolo_tiny_demo.py
│   │
│   └── tests/
│       ├── test_protocol.py
│       ├── test_gemm_correctness.py
│       ├── test_model_lowering.py
│       └── test_quantization.py
│
├── ps/
│   ├── include/
│   │   ├── bcuda_runtime.h
│   │   ├── bcuda_buffer.h
│   │   ├── bcuda_graph_executor.h
│   │   ├── bcuda_ops.h
│   │   └── bcuda_pl_driver.h
│   │
│   ├── src/
│   │   ├── main.c
│   │   ├── protocol_server.c
│   │   ├── command_dispatch.c
│   │   ├── buffer_manager.c
│   │   ├── graph_executor.c
│   │   ├── ops_im2col.c
│   │   ├── ops_activation.c
│   │   ├── ops_pool.c
│   │   ├── ops_concat.c
│   │   ├── ops_detect.c
│   │   ├── ops_nms.c
│   │   ├── pl_driver.c
│   │   └── error.c
│   │
│   └── tests/
│       ├── test_buffer_manager.c
│       ├── test_protocol_decode.c
│       └── test_graph_executor.c
│
├── pl/
│   ├── rtl/
│   │   ├── gemm_accelerator_top.sv
│   │   ├── axi_lite_control.sv
│   │   ├── gemm_controller.sv
│   │   ├── address_generator.sv
│   │   ├── tile_loader.sv
│   │   ├── tile_writer.sv
│   │   ├── tile_cache.sv
│   │   ├── systolic_array_16x16.sv
│   │   ├── pe_int8_mac.sv
│   │   ├── accumulator_buffer.sv
│   │   └── postprocess.sv
│   │
│   ├── hls/
│   │   ├── gemm_hls.cpp
│   │   ├── gemm_hls.h
│   │   └── testbench.cpp
│   │
│   ├── sim/
│   │   ├── tb_pe.sv
│   │   ├── tb_systolic_array.sv
│   │   ├── tb_gemm_top.sv
│   │   └── vectors/
│   │       ├── A.hex
│   │       ├── B.hex
│   │       └── C_ref.hex
│   │
│   └── vivado/
│       ├── block_design.tcl
│       ├── constraints.xdc
│       └── build_bitstream.tcl
│
├── models/
│   ├── mlp/
│   ├── cnn/
│   └── yolo_tiny/
│
├── benchmarks/
│   ├── bench_gemm.py
│   ├── bench_transfer.py
│   ├── bench_cnn.py
│   └── bench_end_to_end.py
│
└── scripts/
    ├── build_host.sh
    ├── build_ps.sh
    ├── build_pl.sh
    ├── deploy_to_zynq.sh
    └── run_demo.sh
```

---

## 4. Common Shared Files

The `common/` directory contains binary protocol and graph definitions shared between host Python and PS C/C++.

### 4.1 `common/bcuda_dtypes.h` and `common/bcuda_dtypes.py`

Purpose:

```text
Define shared numeric constants for data types, tensor layouts, buffer usages, command IDs, status codes, and operation types.
```

Required constants:

```c
typedef enum {
    BC_DTYPE_INT8  = 1,
    BC_DTYPE_INT16 = 2,
    BC_DTYPE_INT32 = 3
} BcudaDtype;

typedef enum {
    BC_LAYOUT_ROW_MAJOR = 1,
    BC_LAYOUT_NCHW      = 2,
    BC_LAYOUT_NHWC      = 3
} BcudaLayout;

typedef enum {
    BC_USAGE_WEIGHT     = 1,
    BC_USAGE_ACTIVATION = 2,
    BC_USAGE_WORKSPACE  = 3,
    BC_USAGE_OUTPUT     = 4,
    BC_USAGE_DESCRIPTOR = 5
} BcudaBufferUsage;
```

Command IDs:

```c
typedef enum {
    BC_CMD_ALLOC   = 1,
    BC_CMD_WRITE   = 2,
    BC_CMD_READ    = 3,
    BC_CMD_EXECUTE = 4,
    BC_CMD_FREE    = 5
} BcudaCommand;
```

Status codes:

```c
typedef enum {
    BC_STATUS_OK                = 0,
    BC_STATUS_GENERAL_ERROR     = 1,
    BC_STATUS_INVALID_COMMAND   = 2,
    BC_STATUS_INVALID_HANDLE    = 3,
    BC_STATUS_OUT_OF_MEMORY     = 4,
    BC_STATUS_SHAPE_MISMATCH    = 5,
    BC_STATUS_UNSUPPORTED_DTYPE = 6,
    BC_STATUS_EXECUTION_FAILED  = 7
} BcudaStatus;
```

Operation types used inside `EXECUTE`:

```c
typedef enum {
    BC_OP_GEMM          = 1,
    BC_OP_GRAPH         = 2,
    BC_OP_CONV2D_GEMM   = 3,
    BC_OP_ACTIVATION    = 4,
    BC_OP_POOL          = 5,
    BC_OP_UPSAMPLE      = 6,
    BC_OP_CONCAT        = 7,
    BC_OP_RESHAPE       = 8,
    BC_OP_DETECT_DECODE = 9,
    BC_OP_NMS           = 10
} BcudaOpType;
```

### 4.2 `common/bcuda_protocol.h` and `common/bcuda_protocol.py`

Purpose:

```text
Define packet headers and request/response structures for ALLOC, WRITE, READ, EXECUTE, and FREE.
```

Common command header:

```c
typedef struct {
    uint32_t magic;          // 0x42435544 = "BCUD"
    uint16_t version;        // protocol version
    uint16_t command;        // BC_CMD_ALLOC, etc.

    uint32_t request_id;     // host-chosen ID
    uint32_t header_bytes;   // full request header size
    uint32_t payload_bytes;  // raw bytes following header

    uint32_t flags;
    uint32_t checksum;       // 0 for MVP
} BcudaHeader;
```

Common response header:

```c
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t command;

    uint32_t request_id;
    uint32_t status;
    uint32_t header_bytes;
    uint32_t payload_bytes;

    uint32_t error_code;
    uint32_t reserved;
} BcudaResponseHeader;
```

Tensor descriptor:

```c
typedef struct {
    uint32_t dtype;
    uint32_t rank;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t dim3;
    uint32_t layout;
    uint32_t reserved;
} BcudaTensorDesc;
```

ALLOC request:

```c
typedef struct {
    BcudaHeader header;

    uint64_t size_bytes;
    BcudaTensorDesc tensor;
    uint32_t usage;
    uint32_t alignment;
} BcudaAllocRequest;
```

ALLOC response:

```c
typedef struct {
    BcudaResponseHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t size_bytes;
} BcudaAllocResponse;
```

WRITE request:

```c
typedef struct {
    BcudaHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t offset_bytes;
    uint64_t write_bytes;

    // Followed by raw payload bytes.
} BcudaWriteRequest;
```

WRITE response:

```c
typedef struct {
    BcudaResponseHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t written_bytes;
} BcudaWriteResponse;
```

READ request:

```c
typedef struct {
    BcudaHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t offset_bytes;
    uint64_t read_bytes;
} BcudaReadRequest;
```

READ response:

```c
typedef struct {
    BcudaResponseHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
    uint64_t read_bytes;

    // Followed by raw payload bytes.
} BcudaReadResponse;
```

EXECUTE request:

```c
typedef struct {
    BcudaHeader header;

    uint32_t op_type;
    uint32_t descriptor_bytes;

    // Followed by operation-specific descriptor.
} BcudaExecuteRequest;
```

EXECUTE response:

```c
typedef struct {
    BcudaResponseHeader header;

    uint32_t op_type;
    uint32_t execution_status;
    uint64_t elapsed_cycles;
} BcudaExecuteResponse;
```

FREE request:

```c
typedef struct {
    BcudaHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
} BcudaFreeRequest;
```

FREE response:

```c
typedef struct {
    BcudaResponseHeader header;

    uint32_t buffer_id;
    uint32_t reserved;
} BcudaFreeResponse;
```

### 4.3 `common/bcuda_graph.h` and `common/bcuda_graph.py`

Purpose:

```text
Define GEMM descriptors, graph descriptors, and layer descriptors.
```

GEMM execution descriptor:

```c
typedef struct {
    uint32_t A_buffer;
    uint32_t B_buffer;
    uint32_t C_buffer;

    uint32_t M;
    uint32_t N;
    uint32_t K;

    uint32_t A_dtype;
    uint32_t B_dtype;
    uint32_t C_dtype;

    uint32_t A_layout;
    uint32_t B_layout;
    uint32_t C_layout;

    uint32_t flags;
    uint32_t reserved;
} BcudaGemmDesc;
```

Graph execution descriptor:

```c
typedef struct {
    uint32_t model_desc_buffer;
    uint32_t input_buffer;
    uint32_t output_buffer;

    uint32_t flags;
    uint32_t reserved;
} BcudaGraphExecDesc;
```

Model descriptor:

```c
typedef struct {
    uint32_t magic;              // "BCMD"
    uint32_t version;

    uint32_t num_layers;
    uint32_t layer_desc_offset;

    uint32_t num_buffers;
    uint32_t buffer_table_offset;

    uint32_t input_buffer;
    uint32_t output_buffer;

    uint32_t activation_arena_a;
    uint32_t activation_arena_b;
    uint32_t workspace_buffer;

    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
} BcudaModelDesc;
```

Model descriptor byte format rules:

```text
- All fields are little-endian unsigned 32-bit integers.
- BcudaModelDesc is 56 bytes.
- magic must be 0x444D4342 ("BCMD" in little-endian memory).
- version must be 1.
- buffer_table_offset points to an array of BcudaModelBufferDesc records.
- layer_desc_offset points to an array of BcudaLayerDesc records.
- input_buffer, output_buffer, activation_arena_a, activation_arena_b, and
  workspace_buffer are buffer-table indexes, not host-visible physical
  addresses.
```

Model buffer descriptor:

```c
typedef struct {
    uint32_t buffer_id;       // Device buffer handle allocated by ALLOC.
    uint32_t dtype;           // BcudaDtype.
    uint32_t layout;          // BcudaLayout.
    uint32_t usage;           // BcudaBufferUsage.
    uint32_t size_bytes;
    uint32_t rank;
    uint32_t dim0;
    uint32_t dim1;
} BcudaModelBufferDesc;
```

Buffer descriptor byte format rules:

```text
- BcudaModelBufferDesc is 32 bytes.
- The current MVP stores dim0 and dim1 because executable PL GEMM is 2D.
- Higher-rank CNN tensors are represented by the layer descriptor fields
  until the descriptor format is extended with dim2/dim3.
```

Layer descriptor:

```c
typedef struct {
    uint32_t op_type;

    uint32_t input0;
    uint32_t input1;
    uint32_t input2;
    uint32_t output;

    uint32_t weight;
    uint32_t bias;
    uint32_t scale;

    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;

    uint32_t out_c;
    uint32_t out_h;
    uint32_t out_w;

    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;

    uint32_t groups;
    uint32_t flags;
    uint32_t reserved;
} BcudaLayerDesc;
```

Layer descriptor byte format rules:

```text
- BcudaLayerDesc is 96 bytes.
- input0, input1, input2, output, weight, bias, and scale are buffer-table
  indexes.
- For BC_OP_GEMM:
    A_buffer = buffer_table[input0].buffer_id
    B_buffer = buffer_table[weight].buffer_id
    C_buffer = buffer_table[output].buffer_id
    M = n
    N = out_c
    K = c
    A/B dtype = INT8
    C dtype = INT32
    layout = ROW_MAJOR
- For BC_OP_CONV2D_GEMM in the MVP firmware path:
    input0 is INT8 NCHW activation
    weight is INT8 OIHW weights
    optional bias is INT32 with out_c entries
    output is INT32 NCHW activation
    n,c,h,w,out_c,out_h,out_w,kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w
    define the convolution geometry
    groups must be 1
    the PS fallback computes the convolution directly until tiled im2col/GEMM
    lowering is implemented
- For BC_OP_ACTIVATION:
    input0 and output must have the same dtype and byte count
    MVP flags=0 means ReLU
    supported dtypes are INT8 and INT32
- For BC_OP_POOL:
    MVP implements INT8 NCHW max-pooling
    n,c,h,w,out_h,out_w,kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w
    define the pooling geometry
- For BC_OP_UPSAMPLE:
    MVP implements nearest-neighbor NCHW resize
    input0 and output must have the same dtype
    n,c,h,w describe the input tensor
    n,c,out_h,out_w describe the output tensor
    supported dtypes are INT8, INT16, and INT32
- For BC_OP_CONCAT:
    MVP implements channel-axis NCHW concat for one to three inputs
    input0 is required; input1 and input2 are optional when nonzero
    output dtype must match all input dtypes
    n,h,w define the shared spatial geometry
    out_c is the total output channel count
    each input channel count is inferred from its buffer byte size
- For BC_OP_RESHAPE:
    input0 and output must have equal byte size
    the operation is a copy unless input and output are the same buffer
- For unsupported op_type values, firmware must return
  BC_STATUS_INVALID_COMMAND until that op's semantics are implemented.
```

---

## 5. Host Computer Part

### 5.1 Host Responsibilities

The host is responsible for:

```text
- User-facing Python API
- Device connection management
- Protocol serialization/deserialization
- Model import
- Quantization
- Model lowering
- Model package construction
- Weight upload
- Input upload
- Output download
- Correctness testing
- Benchmark scripts
```

The host is not responsible for:

```text
- Physical address management
- DMA/cache coherency
- AXI register access
- PL tile scheduling
- Intermediate activation transfers during graph execution
```

### 5.2 `host/budgetcuda/device.py`

Purpose:

```text
High-level device API for ALLOC, WRITE, READ, EXECUTE, and FREE.
```

Required class:

```python
class Device:
    def __init__(self, address: str, port: int = 9000):
        pass

    def alloc(self, shape, dtype, layout="row_major", usage="activation"):
        pass

    def write(self, tensor, np_array, offset_bytes=0):
        pass

    def read(self, tensor, offset_bytes=0, nbytes=None):
        pass

    def execute(self, op_desc):
        pass

    def execute_gemm(self, A, B, C, M, N, K, out_dtype="int32"):
        pass

    def free(self, tensor):
        pass
```

Implementation notes:

```text
- Calls ProtocolClient internally.
- Returns Tensor objects for allocated buffers.
- Validates host-side shapes and dtypes before sending commands.
```

### 5.3 `host/budgetcuda/tensor.py`

Purpose:

```text
Represent a device-resident buffer visible to the host through a logical handle.
```

Required class:

```python
class Tensor:
    def __init__(self, buffer_id, shape, dtype, layout, size_bytes, device):
        self.buffer_id = buffer_id
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.size_bytes = size_bytes
        self.device = device
```

Rules:

```text
- Tensor does not own host memory.
- Tensor does not expose physical addresses.
- Tensor can be read back using device.read(tensor).
```

### 5.4 `host/budgetcuda/protocol.py`

Purpose:

```text
Implement the transport client and command methods.
```

Required class:

```python
class ProtocolClient:
    def alloc(self, size_bytes, tensor_desc, usage, alignment):
        pass

    def write(self, buffer_id, data: bytes, offset_bytes=0):
        pass

    def read(self, buffer_id, offset_bytes, read_bytes) -> bytes:
        pass

    def execute(self, op_type, descriptor: bytes):
        pass

    def free(self, buffer_id):
        pass
```

Implementation notes:

```text
- Uses TCP sockets for final version.
- Can use local mock backend for testing.
- Must check magic, version, request_id, status, and payload size.
```

### 5.5 `host/budgetcuda/serializer.py`

Purpose:

```text
Pack and unpack binary protocol structures.
```

Required functions:

```python
def pack_header(...):
    pass

def pack_tensor_desc(...):
    pass

def pack_gemm_desc(...):
    pass

def pack_graph_exec_desc(...):
    pass

def unpack_response_header(data):
    pass
```

Rules:

```text
- All structures are little-endian.
- Python struct formats must match the C structs exactly.
- Unit tests must validate byte size and field offsets.
```

### 5.6 `host/budgetcuda/model.py`

Purpose:

```text
Represent a model stored on the Zynq device.
```

Required class:

```python
class Model:
    def __init__(self, device, descriptor_tensor, input_tensor, output_tensor, weights):
        self.device = device
        self.descriptor_tensor = descriptor_tensor
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.weights = weights

    def run(self, input_np):
        self.device.write(self.input_tensor, input_np)
        self.device.execute_graph(
            self.descriptor_tensor,
            self.input_tensor,
            self.output_tensor
        )
        return self.device.read(self.output_tensor)
```

### 5.7 `host/budgetcuda/graph.py`

Purpose:

```text
Host-side representation of a Budget-CUDA model graph before serialization.
```

Required classes:

```python
class Graph:
    def __init__(self):
        self.layers = []
        self.buffers = []

class Layer:
    def __init__(self, op_type, inputs, output, params):
        pass
```

Supported layer op types:

```text
OP_GEMM
OP_CONV2D_GEMM
OP_ACTIVATION
OP_POOL
OP_UPSAMPLE
OP_CONCAT
OP_RESHAPE
OP_DETECT_DECODE
OP_NMS
```

### 5.8 `host/budgetcuda/quantization.py`

Purpose:

```text
Quantize FP32 weights and activations to INT8-compatible tensors.
```

MVP support:

```text
- signed symmetric INT8 quantization
- zero point = 0
- INT32 accumulation
```

Required functions:

```python
def quantize_symmetric_int8(x):
    # returns int8_tensor, scale
    pass

def dequantize_int8(x_int8, scale):
    pass
```

Recommended quantization convention:

```text
X_int8 = round(X_fp32 / scale_x)
W_int8 = round(W_fp32 / scale_w)
acc_int32 = X_int8 x W_int8
Y_fp32 approximated by acc_int32 * scale_x * scale_w
```

### 5.9 `host/budgetcuda/converter/`

Purpose:

```text
Convert external models into Budget-CUDA model packages.
```

Files:

```text
torch_import.py:
    Import simple PyTorch models.

onnx_import.py:
    Optional ONNX subset importer.

yolo_import.py:
    Convert a fixed YOLO-like INT8 architecture into Budget-CUDA graph descriptors.

memory_plan.py:
    Assign persistent weight buffers, activation arenas, workspace buffers, and output buffers.
```

YOLO-like model support should be implemented as graph lowering, not as new protocol commands.

---

## 6. Zynq PS Part

### 6.1 PS Responsibilities

The PS is responsible for:

```text
- Receiving protocol packets
- Sending protocol responses
- Allocating Zynq DRAM buffers
- Maintaining buffer_id -> address mappings
- Copying WRITE payloads into DRAM
- Reading DRAM buffers for READ responses
- Executing graph descriptors
- Running non-GEMM operations in software
- Programming PL registers for GEMM
- Handling cache flush/invalidate
- Waiting for PL completion
- Returning execution status
```

### 6.2 `ps/src/main.c`

Purpose:

```text
Start the PS runtime server.
```

Responsibilities:

```text
- Initialize runtime state
- Initialize buffer manager
- Initialize PL driver
- Start TCP/listening server
- Dispatch incoming commands
```

Pseudo-flow:

```c
int main(void) {
    bcuda_runtime_init();
    bcuda_buffer_manager_init();
    bcuda_pl_driver_init();
    bcuda_protocol_server_start(9000);
    return 0;
}
```

### 6.3 `ps/src/protocol_server.c`

Purpose:

```text
Receive requests from host and send responses.
```

Responsibilities:

```text
- Accept socket connection
- Read command header
- Validate magic/version/header size
- Read command-specific payload
- Pass command to command_dispatch
- Send response header and payload
```

### 6.4 `ps/src/command_dispatch.c`

Purpose:

```text
Map protocol command IDs to handler functions.
```

Required handlers:

```c
int handle_alloc(const BcudaAllocRequest *req, BcudaAllocResponse *res);
int handle_write(const BcudaWriteRequest *req, const uint8_t *payload, BcudaWriteResponse *res);
int handle_read(const BcudaReadRequest *req, BcudaReadResponse *res, uint8_t **payload);
int handle_execute(const BcudaExecuteRequest *req, const uint8_t *desc, BcudaExecuteResponse *res);
int handle_free(const BcudaFreeRequest *req, BcudaFreeResponse *res);
```

### 6.5 `ps/src/buffer_manager.c`

Purpose:

```text
Manage the 500 MB Zynq DRAM region as device memory.
```

Required data structure:

```c
typedef struct {
    uint32_t buffer_id;
    uintptr_t physical_address;
    void *virtual_address;
    uint64_t size_bytes;
    uint32_t dtype;
    uint32_t layout;
    uint32_t usage;
    uint32_t rank;
    uint32_t shape[4];
    uint32_t state;
} BcudaBuffer;
```

Required functions:

```c
int bcuda_alloc_buffer(uint64_t size_bytes, BcudaTensorDesc desc, uint32_t usage, uint32_t *out_id);
int bcuda_free_buffer(uint32_t buffer_id);
int bcuda_get_buffer(uint32_t buffer_id, BcudaBuffer **out);
int bcuda_write_buffer(uint32_t buffer_id, uint64_t offset, const void *src, uint64_t bytes);
int bcuda_read_buffer(uint32_t buffer_id, uint64_t offset, void *dst, uint64_t bytes);
```

Rules:

```text
- Buffer IDs are opaque to the host.
- Buffers must be aligned for DMA/AXI access.
- Overlapping allocations are forbidden.
- Out-of-range WRITE/READ must return an error.
```

### 6.6 `ps/src/graph_executor.c`

Purpose:

```text
Execute a model graph stored in device DRAM.
```

Required function:

```c
int bcuda_execute_graph(const BcudaGraphExecDesc *desc);
```

Execution flow:

```text
1. Resolve model_desc_buffer.
2. Parse BcudaModelDesc.
3. Locate layer descriptors.
4. For each layer:
       if layer.op_type is GEMM or CONV2D_GEMM:
           call GEMM path
       else:
           call PS fallback op
5. Return final status.
```

Pseudo-code:

```c
int bcuda_execute_graph(const BcudaGraphExecDesc *desc) {
    BcudaModelDesc *model = map_model_desc(desc->model_desc_buffer);
    BcudaLayerDesc *layers = get_layer_array(model);

    for (uint32_t i = 0; i < model->num_layers; i++) {
        switch (layers[i].op_type) {
            case BC_OP_GEMM:
                bcuda_op_gemm(&layers[i]);
                break;
            case BC_OP_CONV2D_GEMM:
                bcuda_op_conv2d_gemm(&layers[i]);
                break;
            case BC_OP_ACTIVATION:
                bcuda_op_activation(&layers[i]);
                break;
            case BC_OP_POOL:
                bcuda_op_pool(&layers[i]);
                break;
            case BC_OP_UPSAMPLE:
                bcuda_op_upsample(&layers[i]);
                break;
            case BC_OP_CONCAT:
                bcuda_op_concat(&layers[i]);
                break;
            case BC_OP_DETECT_DECODE:
                bcuda_op_detect_decode(&layers[i]);
                break;
            case BC_OP_NMS:
                bcuda_op_nms(&layers[i]);
                break;
            default:
                return BC_STATUS_INVALID_COMMAND;
        }
    }

    return BC_STATUS_OK;
}
```

### 6.7 `ps/src/ops_gemm.c`

Purpose:

```text
Execute GEMM layers by invoking the PL driver.
```

Responsibilities:

```text
- Convert layer descriptor to GEMM descriptor
- Resolve input/weight/output buffers
- Validate dimensions and dtypes
- Call bcuda_run_gemm()
```

### 6.8 `ps/src/ops_im2col.c`

Purpose:

```text
Lower convolution input feature maps into GEMM-compatible matrix tiles.
```

MVP version:

```text
- Explicit im2col into workspace buffer
```

Final preferred version:

```text
- Tiled im2col to avoid materializing very large im2col matrices
```

### 6.9 `ps/src/ops_activation.c`

Purpose:

```text
Run activation functions on PS when not fused into PL.
```

Required initial support:

```text
- ReLU
- LeakyReLU
```

Optional support:

```text
- SiLU approximation for YOLO-like models
```

### 6.10 `ps/src/ops_pool.c`

Purpose:

```text
Implement max-pooling or average-pooling for CNN models.
```

### 6.11 `ps/src/ops_concat.c`

Purpose:

```text
Concatenate tensors along a selected axis, especially channel-axis concat for YOLO neck layers.
```

### 6.12 `ps/src/ops_detect.c`

Purpose:

```text
Decode raw detector head outputs into bounding box candidates.
```

Responsibilities:

```text
- Convert raw model output to boxes/classes/scores
- Apply confidence threshold
- Write compact detection candidates to output buffer
```

### 6.13 `ps/src/ops_nms.c`

Purpose:

```text
Perform non-maximum suppression on decoded boxes.
```

NMS stays on PS because it is control-heavy and irregular.

### 6.14 `ps/src/pl_driver.c`

Purpose:

```text
Control the PL GEMM accelerator through AXI-Lite registers.
```

Required function:

```c
int bcuda_run_gemm(
    BcudaBuffer *A,
    BcudaBuffer *B,
    BcudaBuffer *C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t flags
);
```

Register write sequence:

```text
1. Flush A and B buffers if required.
2. Invalidate C buffer if required.
3. Write A_ADDR.
4. Write B_ADDR.
5. Write C_ADDR.
6. Write M.
7. Write N.
8. Write K.
9. Write FLAGS.
10. Write CTRL.START.
11. Wait for STATUS.DONE or interrupt.
12. Check STATUS.ERROR.
13. Invalidate C buffer after writeback.
```

---

## 7. Zynq PL Part

### 7.1 PL Responsibilities

The PL implements the fixed-function INT8 GEMM accelerator.

It handles:

```text
- Loading A/B tiles from Zynq DRAM
- Staging tiles into BRAM/SRAM-style caches
- Running 8 parallel 16x16 systolic arrays
- INT8 x INT8 multiplication
- INT32 accumulation
- Writing C tiles back to Zynq DRAM
- Reporting done/error status
```

The PL does not handle:

```text
- Python
- CUDA-like language parsing
- YOLO-specific logic
- model graph scheduling
- memory allocation
- buffer handles
- quantization calibration
```

### 7.2 Hardware Data Format

MVP data format:

```text
Input A: signed INT8
Input B: signed INT8
Output C: signed INT32
Accumulator: signed INT32
Tensor layout: row-major 2D
Shape restriction: M, N, K multiples of 16
```

Final optional data format:

```text
Input A: signed INT8
Input B: signed INT8
Accumulator: signed INT32
Output C: signed INT8 after bias/requantization/activation
```

### 7.3 PL Block Diagram

```text
Zynq DDR
   |
   v
AXI read / DMA
   |
   v
A/B tile caches
   |
   v
8 x 16x16 INT8 systolic arrays
   |
   v
INT32 accumulators
   |
   v
optional bias/requantize/activation
   |
   v
AXI write / DMA
   |
   v
Zynq DDR
```

### 7.4 `pl/rtl/gemm_accelerator_top.sv`

Purpose:

```text
Top-level PL accelerator module.
```

Submodules:

```text
axi_lite_control
gemm_controller
address_generator
tile_loader
tile_writer
tile_cache
8 x systolic_array_16x16
accumulator_buffer
optional postprocess
```

Top-level interfaces:

```text
AXI-Lite slave for control registers
AXI master or DMA-facing interface for memory movement
Clock/reset
Optional interrupt output
```

### 7.5 `pl/rtl/axi_lite_control.sv`

Purpose:

```text
Expose control/status registers to the PS.
```

Minimum register map:

```text
CTRL
STATUS
A_ADDR
B_ADDR
C_ADDR
M
N
K
FLAGS
```

Better final register map:

```text
CTRL
STATUS
A_ADDR_LOW
A_ADDR_HIGH
B_ADDR_LOW
B_ADDR_HIGH
C_ADDR_LOW
C_ADDR_HIGH
M
N
K
A_STRIDE
B_STRIDE
C_STRIDE
FLAGS
ERROR_CODE
CYCLE_COUNT
```

### 7.6 `pl/rtl/gemm_controller.sv`

Purpose:

```text
Control tiled GEMM execution.
```

Responsibilities:

```text
- Read M/N/K configuration
- Iterate over output tiles
- Clear accumulators
- Request A/B tile loads
- Start systolic arrays
- Wait for compute completion
- Request C tile writeback
- Set done/status
```

MVP tile loop:

```text
for m = 0 to M step 16:
    for n_group = 0 to N step 8*16:
        clear accumulators
        for k = 0 to K step Kt:
            load A tile
            load 8 B tiles
            run 8 systolic arrays
        write 8 C tiles
```

### 7.7 `pl/rtl/address_generator.sv`

Purpose:

```text
Generate DDR addresses for A, B, and C tiles.
```

Inputs:

```text
A base address
B base address
C base address
M
N
K
current m/n/k tile indices
strides if supported
```

Outputs:

```text
A tile read address
B tile read address
C tile write address
```

### 7.8 `pl/rtl/tile_loader.sv`

Purpose:

```text
Load A and B tiles from Zynq DRAM into BRAM/SRAM tile caches.
```

Responsibilities:

```text
- Issue read requests
- Receive memory data
- Store into tile caches
- Signal tile_load_done
```

### 7.9 `pl/rtl/tile_writer.sv`

Purpose:

```text
Write output C tiles from accumulator/output buffer back to Zynq DRAM.
```

Responsibilities:

```text
- Read accumulated C tile
- Format output as INT32 or INT8
- Issue memory write requests
- Signal write_done
```

### 7.10 `pl/rtl/tile_cache.sv`

Purpose:

```text
BRAM/SRAM-style tile buffer for A and B matrix tiles.
```

Recommended initial design:

```text
Shared A tile buffer
8 separate B tile buffers
```

Example memory requirement for Kt = 64:

```text
A buffer: 16 x 64 x 1 byte = 1024 bytes
B buffers: 8 x 64 x 16 x 1 byte = 8192 bytes
Total input tile cache = 9216 bytes
```

Optional final optimization:

```text
Ping-pong double buffering for overlapping load and compute.
```

### 7.11 `pl/rtl/systolic_array_16x16.sv`

Purpose:

```text
Compute one 16x16 output tile using 256 processing elements.
```

Each systolic array computes:

```text
C_tile[16x16] = A_tile[16xK] x B_tile[Kx16]
```

Data type:

```text
INT8 input
INT32 accumulator
```

### 7.12 `pl/rtl/pe_int8_mac.sv`

Purpose:

```text
Single processing element for systolic array.
```

Operation:

```verilog
acc <= acc + a_in * b_in;
a_out <= a_in;
b_out <= b_in;
```

Required behavior:

```text
- signed INT8 multiplication
- signed INT32 accumulation
- accumulator clear
- valid/enable gating
- forward A and B values
```

### 7.13 `pl/rtl/accumulator_buffer.sv`

Purpose:

```text
Store output accumulators for all 8 systolic arrays.
```

Capacity:

```text
8 arrays x 16 x 16 accumulators = 2048 INT32 accumulators
2048 x 4 bytes = 8192 bytes
```

### 7.14 `pl/rtl/postprocess.sv`

Purpose:

```text
Optional final-stage bias, requantization, clamp, and activation.
```

MVP:

```text
Bypass postprocess and write INT32 output.
```

Final preferred path:

```text
INT32 accumulator
    + bias
    -> requantize
    -> clamp to INT8
    -> optional ReLU/LeakyReLU
    -> write INT8 output
```

This reduces DRAM traffic for CNN/YOLO-style models.

---

## 8. Host-to-Device Workflows

### 8.1 Direct GEMM Workflow

Goal:

```text
C = A x B
```

Host sequence:

```text
ALLOC A
ALLOC B
ALLOC C
WRITE A
WRITE B
EXECUTE OP_GEMM
READ C
FREE A/B/C
```

Detailed flow:

```text
Host                               Zynq PS                              Zynq PL
----                               -------                              -------
ALLOC A -------------------------> allocate DDR buffer
<------------------------- A_id

ALLOC B -------------------------> allocate DDR buffer
<------------------------- B_id

ALLOC C -------------------------> allocate DDR buffer
<------------------------- C_id

WRITE A_id + A bytes ------------> copy A into DRAM
WRITE B_id + B bytes ------------> copy B into DRAM

EXECUTE GEMM(A,B,C) -------------> resolve buffer IDs
                                     program PL registers -------------> load tiles
                                                                      compute GEMM
                                                                      write C
                                                                      set done
<------------------------- OK

READ C_id -----------------------> read C from DRAM
<------------------------- C bytes
```

### 8.2 Model Loading Workflow

Model loading uses ordinary memory commands. There is no special `LOAD_MODEL` command.

For each weight tensor:

```text
ALLOC weight buffer
WRITE weight bytes
```

For each bias/scale tensor:

```text
ALLOC tensor buffer
WRITE tensor bytes
```

For runtime memory:

```text
ALLOC input buffer
ALLOC output buffer
ALLOC activation arena A
ALLOC activation arena B
ALLOC workspace buffer
```

For graph metadata:

```text
ALLOC model descriptor buffer
WRITE model descriptor bytes
```

The host then stores:

```text
model.descriptor_buffer
model.input_buffer
model.output_buffer
model.weight_buffers
```

### 8.3 Model Inference Workflow

Runtime inference sequence:

```text
WRITE input buffer
EXECUTE OP_GRAPH
READ output buffer
```

Detailed flow:

```text
Host:
    writes input image/tensor
    executes graph using descriptor buffer
    reads final output

PS:
    reads graph descriptor
    executes each layer
    calls PL for GEMM layers
    runs non-GEMM layers locally
    leaves intermediate activations in DRAM

PL:
    executes GEMM calls only
```

---

## 9. CNN and YOLO Support

### 9.1 Generic Model Support Rule

Do not hardcode YOLO as a separate protocol feature.

Support generic graph layers:

```text
CONV2D_GEMM
GEMM
ACTIVATION
POOL
UPSAMPLE
CONCAT
RESHAPE
DETECT_DECODE
NMS
```

YOLO support emerges from graph lowering.

### 9.2 Convolution Lowering

Current MVP deployment subset:

```text
The host compiler can build Budget-CUDA model packages from graph-builder
calls that target the cached IL descriptor format. The supported deployment
subset is:

- INT8 input tensors
- INT8 resident weights
- INT32 Conv2D/GEMM outputs
- INT32-to-INT8 scalar fixed-point requantization
- Conv2D with groups=1
- ReLU on INT8 or INT32
- ReLU or power-of-two Leaky ReLU on 16 unfolded INT8 lanes in PL
- INT8 maxpool
- nearest-neighbor upsample
- channel-axis NCHW concat
- reshape/copy

The current helper build_tiny_conv_relu_package() emits:

input INT8 NCHW
    -> BC_OP_CONV2D_GEMM
    -> INT32 activation
    -> BC_OP_ACTIVATION / ReLU
    -> INT32 output

`BC_OP_REQUANTIZE` converts an INT32 activation to INT8 with a four-INT32
parameter buffer: `[multiplier, right_shift, input_zero_point,
output_zero_point]`. Its defined result is:

```text
sat_int8((((x - input_zero_point) * multiplier) >>> right_shift) + output_zero_point)
```

The production PL path implements the symmetric-accumulator case
`input_zero_point = 0`. It streams from DDR and packs four INT8 results per
DDR write, so a multi-layer INT8 CNN can feed its next Conv2D/GEMM without a
PS data round trip. Per-channel requantization is the next optimization; the
current descriptor and operator semantics deliberately remain per-tensor.

This is sufficient for end-to-end model package upload, cached descriptor
execution, correctness tests, and sequential INT8 CNN blocks.

### 9.2.1 Array Workload Splitting and Activation

The scalable GEMM controller splits `N` into groups of up to eight adjacent 16-column
tiles. Each group shares one loaded A tile, loads one B tile into each active
array, computes in parallel, and writes each C tile through the shared DDR
path. For a final partial group, `active_arrays = ceil(remaining_N / 16)`.

`BC_OP_ACTIVATION` uses `flags[0]` to select Leaky ReLU and `flags[10:8]` as
the leak right shift. The PL activation vector unfolds 16 INT8 elements into
lanes and evaluates all lanes combinationally: positive values pass through;
negative values select `0` for ReLU or `x >>> shift` for Leaky ReLU. No
multiplier is required for leaks of `2^-N`.
```

A convolution:

```text
Input:   X [C_in, H, W]
Weight:  W [C_out, C_in, K_h, K_w]
Output:  Y [C_out, H_out, W_out]
```

is lowered to GEMM:

```text
W_col: [C_out, C_in * K_h * K_w]
X_col: [C_in * K_h * K_w, H_out * W_out]

Y_col = W_col x X_col
```

Execution:

```text
PS: generate X_col using im2col or tiled-im2col
PL: execute GEMM
PL: requantization; PS or PL: bias and activation according to the graph plan
```

### 9.3 YOLO-Like Graph Example

A YOLO-like graph may contain:

```text
CONV2D_GEMM -> ACTIVATION
CONV2D_GEMM -> ACTIVATION
POOL/SPPF
UPSAMPLE
CONCAT
CONV2D_GEMM
DETECT_DECODE
NMS
```

Execution split:

```text
PL:
    GEMM generated from convolution and linear layers

PS:
    im2col
    activation if not fused
    pool
    upsample
    concat
    detect decode
    NMS
```

### 9.4 Detection Output Format

The final output should be compact detections, not full intermediate tensors.

Recommended output format:

```c
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    uint32_t class_id;
} BcudaDetection;
```

For an INT-only final path, fixed-point fields may be used instead.

---

## 10. MVP Scope

The 4-month MVP should support:

```text
Commands:
    ALLOC
    WRITE
    READ
    EXECUTE
    FREE

EXECUTE op types:
    OP_GEMM only

Dtypes:
    INT8 input
    INT32 output

Tensor layout:
    row-major 2D

Shape restriction:
    M, N, K multiples of 16

Hardware:
    at least one working 16x16 systolic array
    preferably scalable to 8 arrays
```

MVP demo:

```python
A_dev = dev.alloc(A.shape, dtype="int8")
B_dev = dev.alloc(B.shape, dtype="int8")
C_dev = dev.alloc((M, N), dtype="int32")

dev.write(A_dev, A)
dev.write(B_dev, B)
dev.execute_gemm(A_dev, B_dev, C_dev, M, N, K)
C = dev.read(C_dev)

assert np.array_equal(C, A.astype(np.int32) @ B.astype(np.int32))
```

---

## 11. Final Scope

The final project should support:

```text
Commands:
    ALLOC
    WRITE
    READ
    EXECUTE
    FREE

EXECUTE op types:
    OP_GEMM
    OP_GRAPH

Graph layer types:
    GEMM
    CONV2D_GEMM
    ACTIVATION
    POOL
    UPSAMPLE
    CONCAT
    RESHAPE
    DETECT_DECODE
    NMS

Model style:
    INT8 MLP
    INT8 small CNN
    YOLO-like tiny detector, if schedule permits

PL:
    8 x 16x16 INT8 systolic arrays
    INT32 accumulation
    optional postprocess path
```

---

## 12. Development Milestones

### Milestone 1: Protocol and CPU Mock

Deliverables:

```text
- common protocol structs
- host ProtocolClient
- mock PS server
- ALLOC/WRITE/READ/FREE working in software
- EXECUTE OP_GEMM running NumPy reference on mock backend
```

### Milestone 2: Single GEMM Accelerator Path

Deliverables:

```text
- PL 16x16 systolic array simulation
- PS PL driver
- EXECUTE OP_GEMM reaches PL
- correctness vs NumPy
```

### Milestone 3: 8-Array GEMM Engine

Deliverables:

```text
- 8 parallel systolic arrays
- tile caches
- tiled GEMM controller
- resource/timing reports
- benchmark scripts
```

### Milestone 4: Model-Resident Execution

Deliverables:

```text
- persistent weight buffers
- model descriptor format
- EXECUTE OP_GRAPH
- simple MLP graph execution
```

### Milestone 5: CNN Support

Deliverables:

```text
- CONV2D_GEMM layer
- PS-side im2col
- activation and pooling ops
- small CNN demo
```

### Milestone 6: YOLO-Like Support

Deliverables:

```text
- upsample
- concat
- detect decode
- NMS
- tiny YOLO-like INT8 demo
```

---

## 13. Testing Requirements

### 13.1 Host Tests

Required:

```text
- protocol packing/unpacking tests
- tensor descriptor tests
- GEMM descriptor tests
- quantization tests
- model lowering tests
- mock backend tests
```

### 13.2 PS Tests

Required:

```text
- buffer allocation/free tests
- invalid handle tests
- out-of-range READ/WRITE tests
- command decode tests
- graph descriptor parsing tests
- fallback op correctness tests
```

### 13.3 PL Tests

Required:

```text
- PE testbench
- 16x16 systolic array testbench
- tile loader testbench
- accumulator testbench
- full GEMM top-level testbench
- random matrix vectors compared against software reference
```

### 13.4 End-to-End Tests

Required:

```text
- small GEMM correctness
- large GEMM correctness
- repeated GEMM calls
- persistent weight reuse
- simple MLP inference
- small CNN inference
```

---

## 14. Benchmark Requirements

Measure separately:

```text
- host-to-Zynq WRITE bandwidth
- Zynq-to-host READ bandwidth
- PL GEMM compute latency
- PS fallback op latency
- end-to-end model latency
- compute-only GOPS
- effective end-to-end throughput
```

Do not report only peak theoretical GOPS.

Theoretical peak for 8 arrays:

```text
8 arrays x 16 x 16 PEs = 2048 MACs/cycle

At 100 MHz:
    204.8 GMAC/s
    409.6 GOPS if multiply+add counted separately
```

Actual performance must include:

```text
- tile loading
- writeback
- control overhead
- PS fallback overhead
- host transfer overhead
```

---

## 15. Key Risks

### 15.1 Data Movement Bottleneck

Model residency reduces host-device transfer but not DRAM-PL transfer. Tiling and caching are still required.

Mitigation:

```text
- store weights persistently in Zynq DRAM
- keep intermediate activations in Zynq DRAM
- use tile caches in PL
- add double buffering if time permits
```

### 15.2 PS Bottleneck

If PS handles im2col, activation, requantization, concat, and NMS, it may dominate runtime.

Mitigation:

```text
- fuse bias/requantization/activation into PL after GEMM
- use tiled im2col
- benchmark PS fallback ops separately
```

### 15.3 PL Resource Pressure

8 x 16x16 arrays require 2048 PEs. This may exceed available resources depending on board size and implementation style.

Mitigation:

```text
- build one array first
- scale to 2, 4, then 8 arrays
- reduce array count if timing/resources fail
- use HLS/RTL resource reports early
```

### 15.4 Scope Creep

YOLO support can become too large.

Mitigation:

```text
- complete GEMM path first
- complete MLP second
- complete small CNN third
- treat YOLO-like model as stretch/final demo
```

---

## 16. Final Design Statement

Budget-CUDA is structured as a three-part accelerator stack. The host computer provides a Python API, quantization tools, model conversion, graph serialization, and a protocol client. The Zynq PS acts as the local device runtime: it parses the five-command protocol, manages Zynq DRAM using logical buffer handles, stores quantized model weights and descriptors, executes model graphs, runs non-GEMM operators in software, and invokes the PL for GEMM operations. The PYNQ-Z2 deployment bitstream implements one 16x16 INT8 systolic array with tile caches and INT32 accumulation. The source-level cluster remains scalable to eight arrays for a larger FPGA target; synthesis showed that eight full arrays do not fit the PYNQ-Z2 Zynq-7020 fabric.

The external protocol remains simple and generic:

```text
ALLOC
WRITE
READ
EXECUTE
FREE
```

Model loading, matrix multiplication, CNN inference, and YOLO-like execution are all expressed through these commands using persistent buffers and graph descriptors. This keeps the system implementable while still resembling the memory-resident execution style of modern accelerators.
