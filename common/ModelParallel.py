import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from replicate_KD import replicate_KD
from torch.nn.parallel.parallel_apply import parallel_apply


class ModelParallel(nn.Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, teacher, student, device_ids=None, output_device=None):
        super(DataParallel_KD, self).__init__()

        if not torch.cuda.is_available():
            assert "cuda is not available"
            return

        if device_ids is None:
            assert "there is no device_ids"
        if output_device is None:
            output_device = device_ids[1]
        self.dim = dim
        self.module = [teacher.Gpu(device_ids[0]), student.Gpu(device_ids[1])]
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.to_gpu(device_ids[0])

    def forward(self, *inputs, **kwargs):
        input_t = inputs.copy().cuda(device_ids[0])
        input_s = inputs.copy().cuda(device_ids[1])
        
        outputs = self.parallel_apply(self.module, [input_t, input_s, kwargs)
        return outputs[0].cuda(output_device), outputs[1].cuda(output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
