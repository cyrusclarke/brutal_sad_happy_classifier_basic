import torch

#expected outputs if gpu enabled

In [2]: torch.cuda.current_device()
#Out[2]: 0

#In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x7efce0b03be0>

In [4]: torch.cuda.device_count()
#Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
#Out[5]: 'GeForce GTX 1070'