"""A bounded-SM background stream; foreground stays on the primary context."""
from dataclasses import dataclass
import torch


@dataclass
class WritebackResources:
    stream: torch.cuda.Stream
    requested_sms: int
    actual_sms: int
    total_sms: int
    context: object = None
    descriptor: object = None


_resources = {}


def writeback_resources(device, sm_budget=32):
    device = torch.device(device)
    ordinal = torch.cuda.current_device() if device.index is None else device.index
    key = (ordinal, sm_budget)
    if key in _resources:
        return _resources[key]
    with torch.cuda.device(ordinal):
        total = torch.cuda.get_device_properties(ordinal).multi_processor_count
        if not 0 <= sm_budget <= total:
            raise ValueError(f'writeback SM budget must be between 0 and {total}')
        if not sm_budget:
            result = WritebackResources(torch.cuda.Stream(device=ordinal), 0, total, total)
        else:
            from cuda.bindings import driver

            def checked(call):
                status, *values = call
                if status != driver.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f'CUDA Green Context creation failed: {status}; '
                                       'use --writeback-sm-budget 0 for an unrestricted stream')
                return values[0] if len(values) == 1 else values

            dev = checked(driver.cuDeviceGet(ordinal))
            resource = checked(driver.cuDeviceGetDevResource(dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
            parts, count, _ = checked(driver.cuDevSmResourceSplitByCount(1, resource, 0, sm_budget))
            if count != 1:
                raise RuntimeError('CUDA did not produce one background SM resource')
            descriptor = checked(driver.cuDevResourceGenerateDesc([parts[0]], 1))
            context = checked(driver.cuGreenCtxCreate(descriptor, dev, 1))
            raw = checked(driver.cuGreenCtxStreamCreate(context, 1, 0))
            stream = torch.cuda.ExternalStream(int(raw), device=ordinal)
            result = WritebackResources(stream, sm_budget, parts[0].sm.smCount, total, context, descriptor)
        # Own resources for the process lifetime: captured graphs refer to them.
        _resources[key] = result
        return result
