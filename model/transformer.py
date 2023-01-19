import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor 
import math 
from typing import Tuple 


import torch
import torch.nn as nn 
import math
from torch import nn, Tensor

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)



class TimeSeriesTransformer(nn.Module):

  def __init__(self,
               input_size: int,
               dec_seq_len: int,
               batch_first: bool,
               max_seq_len: int,
               out_seq_len: int=58,
               dim_val: int=512,
               n_encoder_layers: int=4,
               n_decoder_layers: int=4,
               n_heads: int=8,
               dropout_encoder: float=0.2,
               dropout_decoder: float=0.2,
               dropout_pos_enc: float=0.1,
               dim_feedforward_encoder: int=2048,
               dim_feedforward_decoder: int=2048,
               num_predicted_features: int=1
               ):
    
    # super(TimeSeriesTransformer, self).__init__()
    super().__init__()

    self.dec_seq_len = dec_seq_len

    # Creating the three linear layers needed for the model
    self.encoder_input_layer = nn.Linear(
              in_features=input_size, #number of different features, not sequence length
              out_features=dim_val
          )

    self.decoder_input_layer = nn.Linear(
              in_features=num_predicted_features,
              out_features=dim_val
          )

    self.linear_mapping = nn.Linear(
              in_features=dim_val,
              out_features=num_predicted_features
          )

    #Create positional encoder
    self.positional_encoding_layer = PositionalEncoder(
              d_model=dim_val,
              dropout=dropout_pos_enc
          )

    # The encoder layer used in the paper is identical to the one used by
    # Vaswani et al (2017) on which the PyTorch module is based.
    encoder_layer = nn.TransformerEncoderLayer(
              d_model=dim_val,
              nhead=n_heads,
              dim_feedforward=dim_feedforward_encoder,
              dropout=dropout_encoder,
              batch_first=batch_first
          )

    # Stack the encoder layers in nn.TransformerDecoder
    # It seems the option of passing a normalization instance is redundant
    # in my case, because nn.TransformerEncoderLayer per default normalizes
    # after each sub-layer
    # (https://github.com/pytorch/pytorch/issues/24930).
    self.encoder = nn.TransformerEncoder(
              encoder_layer=encoder_layer,
              num_layers=n_encoder_layers,
              norm=None
          )

    decoder_layer = nn.TransformerDecoderLayer(
              d_model=dim_val,
              nhead=n_heads,
              dim_feedforward=dim_feedforward_decoder, 
              dropout=dropout_decoder, 
              batch_first=batch_first
          )

    # Stack the decoder layers in nn.TransformerDecoder
    # It seems the option of passing a normalization instance is redundant
    # in my case, because nn.TransformerDecoderLayer per default normalizes
    # after each sub-layer
    # (https://github.com/pytorch/pytorch/issues/24930).
    self.decoder = nn.TransformerDecoder(
              decoder_layer=decoder_layer,
              num_layers=n_decoder_layers,
              norm=None
          )

  def forward(self,
              src: Tensor,
              tgt: Tensor,
              src_mask: Tensor=None,
              tgt_mask: Tensor=None) -> Tensor:

              """
              Returns a tensor of shape:

              [target_sequence_length, batch_size, num_predicted_features]

              Args:

                  src: the encoder's output sequence (also the input sequence): Shape: (S,E) for unbatched
                  input, (S,N,E) if batch_first=False or (N,S,E) if batch_first=True,
                  where S is the source sequence length, N is the batch size, and
                  E is the number of features (1 if univariate)


                  tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                  (T,N,E) if batch_first=False or (N, T, E) if batch_first=True,
                  where T is the target sequence length, N is the batch size, and
                  E is the number of features (1 if univariate)


                  src_mask: the mask for the src sequence to prevent the model 
                  from using dat points from the target sequence


                  tgt_mask: the mask for the tgt sequence to prevent the model
                  from using data points from the target sequence

              """


              # Pass through the input layer right before the encoder
              src = self.encoder_input_layer(src)
              # src shape: [batch_size, src length, dim_val] regarless of number
              # of input features


              # Pass through the positional encoding layer
              src = self.positional_encoding_layer(src)
              # src shape: [batch_size, src length, dim_val] regardless of number
              # of input features

              # Pass through all the stacked encoder layers in the encoder
              # Padding masking in the encoder is only needed if input sequences
              # are padded (ex. sentences of different length), because all the 
              # sequences are the same length in this time series use case, this
              # type of masking is not needed
              src = self.encoder(src=src)
              # src shape: [batch_size, enc_seq_len, dim_val]


              # Pass decoder input through decoder input layer
              decoder_output = self.decoder_input_layer(tgt)
              # src shape: [target sequence length, batch_size, dim_val]


              # Pass through decoder 
              # Output shape: [batch_size, target seq len, dim_val]
              decoder_output = self.decoder(
                  tgt=decoder_output,
                  memory=src,
                  tgt_mask=tgt_mask,
                  memory_mask=src_mask
              )


              # Pass through linear mapping
              decoder_output = self.linear_mapping(decoder_output)

              return decoder_output
