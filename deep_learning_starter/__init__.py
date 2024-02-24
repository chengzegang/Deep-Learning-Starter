import torch.backends

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.CUDNN_TENSOR_DTYPES.add(torch.bfloat16)
