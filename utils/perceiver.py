import einops
import haiku as hk
import numpy as np
from utils import position_encoding
import jax
import jax.numpy as jnp
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

ModalitySizeT = Mapping[str, int]
PreprocessorOutputT = Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]
PreprocessorT = Callable[..., PreprocessorOutputT]
PostprocessorT = Callable[..., Any]

class ImagePreprocessor(hk.Module):
  """Image preprocessing for Perceiver Encoder."""

  def __init__(
      self,
      prep_type='conv',
      spatial_downsample: int = 4,
      temporal_downsample: int = 1,
      position_encoding_type: str = 'fourier',
      n_extra_pos_mlp: int = 0,
      num_channels: int = 64,
      conv_after_patching: bool = False,
      conv2d_use_batchnorm: bool = True,
      concat_or_add_pos: str = 'concat',
      name: Optional[str] = None,
      **position_encoding_kwargs):
    super().__init__(name=name)

    if prep_type not in ('conv', 'patches', 'pixels', 'conv1x1'):
      raise ValueError('Invalid prep_type!')

    if concat_or_add_pos not in ['concat', 'add']:
      raise ValueError(
          f'Invalid value {concat_or_add_pos} for concat_or_add_pos.')

    self._prep_type = prep_type
    self._spatial_downsample = spatial_downsample
    self._temporal_downsample = temporal_downsample
    self._concat_or_add_pos = concat_or_add_pos
    self._conv_after_patching = conv_after_patching
    self._num_channels = num_channels

    if self._prep_type == 'conv':
      # Downsampling with conv is currently restricted
      convnet_num_layers = math.log(spatial_downsample, 4)
      convnet_num_layers_is_int = (
          convnet_num_layers == np.round(convnet_num_layers))
      if not convnet_num_layers_is_int or temporal_downsample != 1:
        raise ValueError('Only powers of 4 expected for spatial '
                         'and 1 expected for temporal '
                         'downsampling with conv.')

      self.convnet = Conv2DDownsample(
          num_layers=int(convnet_num_layers),
          num_channels=num_channels,
          use_batchnorm=conv2d_use_batchnorm)
    elif self._prep_type == 'conv1x1':
      assert temporal_downsample == 1, 'conv1x1 does not downsample in time.'
      self.convnet_1x1 = hk.Conv2D(
          num_channels, kernel_shape=[1, 1],
          # spatial_downsample is unconstrained for 1x1 convolutions.
          stride=[spatial_downsample, spatial_downsample])

    # Partially construct the positional encoding function.
    # We fully construct it when we know the input size.
    self._positional_encoding_ctor = functools.partial(
        position_encoding.build_position_encoding,
        position_encoding_type=position_encoding_type,
        **position_encoding_kwargs)

    # Stack MLPs to get a deeper positional embedding.
    self._n_extra_pos_mlp = n_extra_pos_mlp

  def _build_network_inputs(
      self, inputs: jnp.ndarray, pos: jnp.ndarray,
      network_input_is_1d: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Construct the final input, including position encoding."""
    batch_size = inputs.shape[0]
    index_dims = inputs.shape[1:-1]

    # Reshape input features to a 1D index dimension if necessary.
    if len(inputs.shape) > 3 and network_input_is_1d:
      inputs = jnp.reshape(
          inputs, [batch_size, np.prod(index_dims), -1])

    # Construct the position encoding.
    pos_enc = self._positional_encoding_ctor(
        index_dims=index_dims)(batch_size=batch_size, pos=pos)

    for i in range(0, self._n_extra_pos_mlp):
      pos_enc += hk.Linear(pos_enc.shape[-1])(pos_enc)
      if i < (self._n_extra_pos_mlp-1):
        pos_enc = jax.nn.relu(pos_enc)

    if not network_input_is_1d:
      # Reshape pos to match the input feature shape
      # if the network takes non-1D inputs
      sh = inputs.shape
      pos_enc = jnp.reshape(pos_enc, list(sh)[:-1]+[-1])

    if self._concat_or_add_pos == 'concat':
      inputs_with_pos = jnp.concatenate([inputs, pos_enc], axis=-1)
    elif self._concat_or_add_pos == 'add':
      inputs_with_pos = inputs + pos_enc

    return inputs_with_pos, inputs

  def __call__(
      self, inputs: jnp.ndarray, *,
      is_training: bool,
      pos: Optional[jnp.ndarray] = None,
      network_input_is_1d: bool = True) -> PreprocessorOutputT:
    if self._prep_type == 'conv':
      # Convnet image featurization.
      # Downsamples spatially by a factor of 4
      conv = self.convnet
      if len(inputs.shape) == 5:
        conv = hk.BatchApply(conv)

      inputs = conv(inputs, is_training=is_training)
    elif self._prep_type == 'conv1x1':
      # maps inputs to 64d

      conv = self.convnet_1x1

      if len(inputs.shape) == 5:
        conv = hk.BatchApply(conv)

      inputs = conv(inputs)
    elif self._prep_type == 'patches':
      # Space2depth featurization.
      # Video: B x T x H x W x C
      inputs = space_to_depth(
          inputs,
          temporal_block_size=self._temporal_downsample,
          spatial_block_size=self._spatial_downsample)

      if inputs.ndim == 5 and inputs.shape[1] == 1:
        # for flow
        inputs = jnp.squeeze(inputs, axis=1)

      if self._conv_after_patching:
        inputs = hk.Linear(self._num_channels, name='patches_linear')(inputs)
    elif self._prep_type == 'pixels':
      # if requested, downsamples in the crudest way
      if inputs.ndim == 4:
        inputs = inputs[:,
                        ::self._spatial_downsample, ::self._spatial_downsample]
      elif inputs.ndim == 5:
        inputs = inputs[:, ::self._temporal_downsample,
                        ::self._spatial_downsample, ::self._spatial_downsample]
      else:
        raise ValueError('Unsupported data format for pixels.')

    inputs, inputs_without_pos = self._build_network_inputs(
        inputs, pos, network_input_is_1d)
    modality_sizes = None  # Size for each modality, only needed for multimodal
    return inputs, modality_sizes, inputs_without_pos
