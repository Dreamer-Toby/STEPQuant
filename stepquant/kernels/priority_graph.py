"""Foreground priorities with native PyTorch graph replay and ownership.

PyTorch 2.11 instantiates CUDA graphs with UseNodePriority. Keep the source
only to set kernel attributes before native instantiation; RNG replay and
allocator lifetime remain managed by PyTorch.
"""
import torch
from cuda.bindings import driver


def _checked(call):
    status,*values=call
    if status!=driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'CUDA graph priority operation failed: {status}')
    return values[0] if len(values)==1 else values


class PriorityDecodeGraph(torch.cuda.CUDAGraph):
    def __new__(cls):
        return super().__new__(cls,keep_graph=True)

    def __init__(self):
        super().__init__(keep_graph=True)

    def capture_end(self):
        super().capture_end()
        self.instantiate()

    def instantiate(self):
        graph=driver.CUgraph(self.raw_cuda_graph())
        _,count=_checked(driver.cuGraphGetNodes(graph))
        nodes,_=_checked(driver.cuGraphGetNodes(graph,count))
        foreground=background=0
        for node in nodes:
            if _checked(driver.cuGraphNodeGetType(node))!=driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                continue
            params=_checked(driver.cuGraphKernelNodeGetParams(node))
            name=_checked(driver.cuFuncGetName(params.func))
            if isinstance(name,bytes):name=name.decode()
            is_background=name=='fit' or name.startswith('fit_') or name.startswith('prepare_rows_kernel')
            # Green Context nodes retain their captured background priority.
            if not is_background:
                value=driver.CUkernelNodeAttrValue()
                value.priority=-1
                _checked(driver.cuGraphKernelNodeSetAttribute(node,driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_PRIORITY,value))
            background+=int(is_background);foreground+=int(not is_background)
        self.priority_counts=(foreground,background)
        super().instantiate()
