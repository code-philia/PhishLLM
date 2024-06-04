import torch
import torch.nn.functional as F
from torch import nn
from clip.model import LayerNorm, OrderedDict, ResidualAttentionBlock, QuickGELU
import torchvision.transforms as transforms
import clip

class QuantizeRelu(nn.Module):
    # Assuming you've defined this somewhere
    def __init__(self, step_size=0.05):
        super(QuantizeRelu, self).__init__()
        self.step_size = step_size

    def forward(self, x):
        return torch.mul(torch.floor(x / self.step_size), self.step_size)


class QuantizeGelu(nn.Module):
    def __init__(self, step_size=0.05):
        super(QuantizeGelu, self).__init__()
        self.step_size = step_size
        self.qrelu = QuantizeRelu(step_size)
        self.qrelu.register_backward_hook(self.zero_gradients_hook)

    def forward(self, x):
        out = self.qrelu(x.half())
        out = out * torch.sigmoid(1.702 * out)
        return out

    def zero_gradients_hook(self, module, grad_input, grad_output):
        # Returns a new gradient tuple with all gradients set to zero
        return tuple(torch.zeros_like(g) if g is not None else None for g in grad_input)


class ResidualAttentionBlockQuantize(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head).to(dtype=torch.half)  # Convert to 'half' data type
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4).to(dtype=torch.half)),  # Convert to 'half' data type
            ("gelu", QuantizeGelu()),
            ("c_proj", nn.Linear(d_model * 4, d_model).to(dtype=torch.half))  # Convert to 'half' data type
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.qrelu = QuantizeRelu()

        # Registering the backward hook to zero out gradients
        self.register_backward_hook(self.zero_gradients_hook)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x.half(), x.half(), x.half(), need_weights=False, attn_mask=self.attn_mask)[
            0]  # Convert to 'half' data type

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = self.qrelu(x.half()) + self.mlp(self.ln_2(x))
        return x

    def zero_gradients_hook(self, module, grad_input, grad_output):
        # Returns a new gradient tuple with all gradients set to zero
        return tuple(torch.zeros_like(g) if g is not None else None for g in grad_input)


def protect_act(model):
    for name, module in model.named_modules():
        if module != model:
            if name == 'gelu':
                setattr(model, name, QuantizeGelu())
            elif isinstance(module, nn.Module) or isinstance(module, nn.Sequential):
                protect_act(module)

def protect_resnetblock(model):
    modules = []
    block_names = []
    for name, module in model.transformer.resblocks.named_modules():
        if isinstance(module, ResidualAttentionBlock):  # Replace with the actual ResNet block class name
            modules.append(module)
            block_names.append(name)

    if modules:
        module = modules[-1]
        name = block_names[-1]
        new_block = ResidualAttentionBlockQuantize(module.attn.embed_dim,
                                                   module.attn.num_heads, None)
        new_block.load_state_dict(module.state_dict())
        new_block.training = module.training
        setattr(model.transformer.resblocks, name, new_block)


def freeze_params(model):
    model.eval()
    model.token_embedding.eval()
    model.transformer.eval()
    model.ln_final.eval()
    model.positional_embedding.requires_grad = False
    model.text_projection.requires_grad = False
    model.logit_scale.requires_grad = False



def print_model(model):
    for name, module in model.named_modules():
        print(module)


def reset_model(path, protect=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(torch.load(path)) # load state_dict
    if device == "cpu":
        model.float()
    if protect: # protect activations functions
        protect_act(model.visual)  #
        protect_resnetblock(model.visual)
    model = model.to(device)
    freeze_params(model)
    return model