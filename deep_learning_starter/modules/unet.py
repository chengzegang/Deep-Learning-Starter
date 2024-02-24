__all__ = [
    "Unet",
    "Unet1d",
    "Unet2d",
    "Unet3d",
    "UnetEncoder",
    "UnetEncoder1d",
    "UnetEncoder2d",
    "UnetEncoder3d",
    "UnetDecoder",
    "UnetDecoder1d",
    "UnetDecoder2d",
    "UnetDecoder3d",
]
from typing import List, Optional, Tuple, Type
import torch
from torch import nn, Tensor
from .transformer import TransformerLayer
from .rmsnorm import SpatialRMSNorm


class UnetConvolution(nn.Module):
    def __init__(
        self,
        convolution_cls: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels, eps=eps, dtype=dtype, device=device)
        self.conv1 = convolution_cls(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.conv2 = convolution_cls(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.conv3 = convolution_cls(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.shorcut = convolution_cls(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, input_embeds: Tensor) -> Tensor:
        residual = self.shorcut(input_embeds)
        hidden_states = self.norm1(input_embeds)
        hidden_states = self.nonlinear(self.conv1(hidden_states)) * self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class SpatialTransformerLayer(TransformerLayer):

    def forward(self, input_embeds: Tensor) -> Tensor:
        """
        Forward pass of the UNet module.

        Args:
            input_embeds (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        shape = input_embeds.shape
        input_embeds = input_embeds.flatten(2).transpose(-1, -2)
        input_embeds = super().forward(input_embeds)
        input_embeds = input_embeds.transpose(-1, -2).view(shape)
        return input_embeds


class UnetEncoderLayer(nn.Module):
    def __init__(
        self,
        convolution_cls: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        unet_shortcut: bool = False,
        attention_layer: bool = True,
        bias: bool = True,
        eps: float = 1e-5,
        num_heads: Optional[int] = 8,
        head_size: Optional[int] = 64,
        max_seq_length: Optional[int] = 8192,
        freq_base: Optional[int] = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.unet_shortcut = unet_shortcut

        self.convolution = UnetConvolution(
            convolution_cls,
            in_channels,
            out_channels,
            bias=bias,
            eps=eps,
            dtype=dtype,
            device=device,
        )
        self.scale_sampler = convolution_cls(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.attention = (
            SpatialTransformerLayer(
                hidden_size=out_channels,
                head_size=head_size,
                num_heads=num_heads,
                max_seq_length=max_seq_length,
                freq_base=freq_base,
                eps=eps,
                dtype=dtype,
                device=device,
            )
            if attention_layer
            else None
        )

    def forward(self, input_embeds: Tensor) -> Tensor:
        input_embeds = self.convolution(input_embeds)
        input_embeds = self.scale_sampler(input_embeds)
        if self.attention is not None:
            input_embeds = self.attention(input_embeds)
        return input_embeds


class UnetDecoderLayer(nn.Module):
    def __init__(
        self,
        convolution_cls: Type[nn.Module],
        convolution_transpose_cls: Type[nn.Module],
        scale_sampler_cls: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        unet_shortcut: bool = False,
        attention_layer: bool = True,
        bias: bool = True,
        eps: float = 1e-5,
        num_heads: Optional[int] = 8,
        head_size: Optional[int] = 64,
        max_seq_length: Optional[int] = 8192,
        freq_base: Optional[int] = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.unet_shortcut = unet_shortcut
        in_channels = in_channels * (1 + self.unet_shortcut)
        self.attention = (
            SpatialTransformerLayer(
                hidden_size=in_channels,
                head_size=head_size,
                num_heads=num_heads,
                max_seq_length=max_seq_length,
                freq_base=freq_base,
                eps=eps,
                dtype=dtype,
                device=device,
            )
            if attention_layer
            else None
        )
        self.scale_sampler = convolution_transpose_cls(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.convolution = UnetConvolution(
            convolution_cls,
            in_channels,
            out_channels,
            bias=bias,
            eps=eps,
            dtype=dtype,
            device=device,
        )

    def forward(self, input_embeds: Tensor, previous_embeds: Optional[Tensor] = None) -> Tensor:
        if self.unet_shortcut:
            assert previous_embeds is not None
            input_embeds = torch.cat([input_embeds, previous_embeds], dim=1)
        if self.attention is not None:
            input_embeds = self.attention(input_embeds)
        input_embeds = self.scale_sampler(input_embeds)
        input_embeds = self.convolution(input_embeds)
        return input_embeds


class UnetEncoder(nn.Module):
    def __init__(
        self,
        convolution_cls: Type[nn.Module],
        in_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.unet_shortcut = unet_shortcut
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(1, num_layers + 1)]

        self.in_conv = convolution_cls(
            in_channels,
            base_channels,
            kernel_size=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                UnetEncoderLayer(
                    convolution_cls=convolution_cls,
                    in_channels=_in_channels[i],
                    out_channels=_out_channels[i],
                    unet_shortcut=False,
                    attention_layer=unet_layer_attention,
                    bias=bias,
                    eps=eps,
                    num_heads=_in_channels[i] // head_size,
                    head_size=head_size,
                    max_seq_length=max_seq_length,
                    freq_base=freq_base,
                    dtype=dtype,
                    device=device,
                )
            )
        self.out_attention = (
            SpatialTransformerLayer(
                hidden_size=_out_channels[-1],
                head_size=head_size,
                num_heads=_out_channels[-1] // head_size,
                max_seq_length=max_seq_length,
                freq_base=freq_base,
                eps=eps,
                dtype=dtype,
                device=device,
            )
            if unet_latent_attention
            else None
        )

        self.out_norm = SpatialRMSNorm(num_features=_out_channels[-1], eps=eps, dtype=dtype, device=device)
        self.out_conv = convolution_cls(
            _out_channels[-1],
            latent_dim,
            kernel_size=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.nonlinear = nn.SiLU(True)

    def _sequential_forward(self, input_embeds: Tensor) -> Tensor:
        embeds = self.in_conv(input_embeds)
        for layer in self.layers:
            embeds = layer(embeds)
        if self.out_attention is not None:
            embeds = self.out_attention(embeds)
        embeds = self.out_norm(embeds)
        embeds = self.nonlinear(embeds)
        embeds = self.out_conv(embeds)
        return embeds

    def _shortcut_forward(self, input_embeds: Tensor) -> Tuple[Tensor, List[Tensor]]:
        embeds = self.in_conv(input_embeds)
        hidden_states = [embeds]
        for layer in self.layers:
            hidden_states.append(layer(hidden_states[-1]))
        latent = hidden_states[-1]
        if self.out_attention is not None:
            latent = self.out_attention(latent)
        latent = self.out_norm(latent)
        latent = self.nonlinear(latent)
        latent = self.out_conv(latent)
        return latent, hidden_states

    def forward(self, input_embeds: Tensor) -> Tuple[Tensor, List[Tensor]] | Tensor:
        if self.unet_shortcut:
            return self._shortcut_forward(input_embeds)
        else:
            return self._sequential_forward(input_embeds)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        convolution_cls: Type[nn.Module],
        convolution_transpose_cls: Type[nn.Module],
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.unet_shortcut = unet_shortcut
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers, 0, -1)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(num_layers - 1, -1, -1)]

        self.layers = nn.ModuleList()

        self.in_conv = convolution_cls(
            latent_dim,
            _in_channels[0],
            kernel_size=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.in_attention = (
            SpatialTransformerLayer(
                hidden_size=_in_channels[0],
                head_size=head_size,
                num_heads=_in_channels[0] // head_size,
                max_seq_length=max_seq_length,
                freq_base=freq_base,
                eps=eps,
                dtype=dtype,
                device=device,
            )
            if unet_latent_attention
            else None
        )
        self.in_norm = SpatialRMSNorm(num_features=_in_channels[0], eps=eps, dtype=dtype, device=device)

        for i in range(num_layers):
            self.layers.append(
                UnetDecoderLayer(
                    convolution_cls=convolution_cls,
                    convolution_transpose_cls=convolution_transpose_cls,
                    scale_sampler_cls=convolution_cls,
                    in_channels=_in_channels[i],
                    out_channels=_out_channels[i],
                    unet_shortcut=unet_shortcut,
                    attention_layer=unet_layer_attention,
                    bias=bias,
                    eps=eps,
                    num_heads=_in_channels[i] // head_size,
                    head_size=head_size,
                    max_seq_length=max_seq_length,
                    freq_base=freq_base,
                    dtype=dtype,
                    device=device,
                )
            )
        self.out_norm = SpatialRMSNorm(_out_channels[-1], eps=eps, dtype=dtype, device=device)
        self.out_conv = convolution_cls(
            _out_channels[-1],
            out_channels,
            kernel_size=1,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.nonlinear = nn.SiLU(True)

    def _sequential_forward(self, latent: Tensor) -> Tensor:
        embeds = self.in_conv(latent)
        if self.in_attention is not None:
            embeds = self.in_attention(embeds)
        embeds = self.in_norm(embeds)
        for layer in self.layers:
            embeds = layer(embeds)
        embeds = self.out_norm(embeds)
        embeds = self.nonlinear(embeds)
        embeds = self.out_conv(embeds)
        return embeds

    def _shortcut_forward(self, latent: Tensor, hidden_states: List[Tensor]) -> Tensor:
        embeds = self.in_conv(latent)
        if self.in_attention is not None:
            embeds = self.in_attention(embeds)
        embeds = self.in_norm(embeds)
        for i, layer in enumerate(self.layers):
            embeds = layer(embeds, hidden_states[i])
        embeds = self.out_norm(embeds)
        embeds = self.nonlinear(embeds)
        embeds = self.out_conv(embeds)
        return embeds

    def forward(self, latent: Tensor, hidden_states: Optional[List[Tensor]] = None) -> Tensor:
        if self.unet_shortcut:
            return self._shortcut_forward(latent, hidden_states)
        else:
            assert hidden_states is None
            return self._sequential_forward(latent)


class Unet(nn.Module):
    def __init__(
        self,
        convolution_cls: Type[nn.Module],
        convolution_transpose_cls: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.unet_shortcut = unet_shortcut
        self.encoder = UnetEncoder(
            convolution_cls,
            in_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )
        self.decoder = UnetDecoder(
            convolution_cls,
            convolution_transpose_cls,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )

    def encode(self, inputs: Tensor) -> Tuple[Tensor, List[Tensor]] | Tensor:

        latent = None
        hidden_states = None

        if self.unet_shortcut:
            latent, hidden_states = self.encoder(inputs)
        else:
            latent = self.encoder(inputs)

        return latent, hidden_states

    def decode(self, latent: Tensor, hidden_states: Optional[List[Tensor]] = None) -> Tensor:
        logits = self.decoder(latent, hidden_states)
        return logits

    def forward(self, inputs: Tensor) -> Tensor:
        latent, hidden_states = self.encode(inputs)
        return self.decode(latent, hidden_states[::-1] if self.unet_shortcut else None)


class UnetEncoder1d(UnetEncoder):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        super().__init__(
            nn.Conv1d,
            in_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class UnetEncoder2d(UnetEncoder):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        super().__init__(
            nn.Conv2d,
            in_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class UnetEncoder3d(UnetEncoder):

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        super().__init__(
            nn.Conv3d,
            in_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class UnetDecoder1d(UnetDecoder):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            nn.Conv1d,
            nn.ConvTranspose1d,
            nn.Conv1d,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class UnetDecoder2d(UnetDecoder):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.Conv2d,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class UnetDecoder3d(UnetDecoder):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            nn.Conv3d,
            nn.ConvTranspose3d,
            nn.Conv3d,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class Unet1d(Unet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        super().__init__(
            nn.Conv1d,
            nn.ConvTranspose1d,
            in_channels,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class Unet2d(Unet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            nn.Conv2d,
            nn.ConvTranspose2d,
            in_channels,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )


class Unet3d(Unet):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        unet_shortcut: bool = False,
        unet_layer_attention: bool = False,
        unet_latent_attention: bool = False,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        bias: bool = True,
        eps: float = 1e-5,
        head_size: int = 64,
        max_seq_length: int = 8192,
        freq_base: int = 10000,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            nn.Conv3d,
            nn.ConvTranspose3d,
            in_channels,
            out_channels,
            latent_dim,
            unet_shortcut,
            unet_layer_attention,
            unet_latent_attention,
            base_channels,
            multiplier,
            num_layers,
            bias,
            eps,
            head_size,
            max_seq_length,
            freq_base,
            dtype,
            device,
        )
