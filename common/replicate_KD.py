import torch.cuda.comm as comm
from net import Net
import time
import copy

def replicate_KD(network, devices):
    from torch.nn.parallel._functions import Broadcast

    
    devices = tuple(devices)
    num_replicas = len(devices)
    
    # parameter copy
    
    params = network.GetParamsReplicas()
    #param_copies = Broadcast(devices)(*params)
    param_copies = Broadcast.apply(devices, *params)
    if len(params) > 0 :
        param_copies = [param_copies[i:i + len(params)]
                        for i in range(0, len(param_copies), len(params))]
    
    buffers = network.GetBuffersReplicas()
    buffer_copies = comm.broadcast_coalesced(buffers, devices)
    
    module_copies = []
    
    for i in range(num_replicas):
        
        net = copy.deepcopy(network)
        net.SetParamsReplicas(param_copies[i])
        net.SetBuffersReplicas(buffer_copies[i])
        module_copies.append(net)
    
    return module_copies
