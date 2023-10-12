import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mulheadatt import MultiheadAttentionLayer
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding





class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_arrivetime_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding_ml = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        self.dec_embedding_ml = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder_arrivetime = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder_ml = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder_depth = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder_aiz = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_aiz = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_lng = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder_ml = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.start_conv = nn.Conv1d(in_channels=seq_len, out_channels=64, kernel_size=9, padding=4, stride=1)
        self.start_conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3, stride=1)
        self.start_conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.start_conv4 = nn.Conv1d(in_channels=64, out_channels=seq_len, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.attention_1 = MultiheadAttentionLayer(d_model, n_heads)
        self.attention_2 = MultiheadAttentionLayer(d_model, n_heads)
        self.projection_lat = nn.Sequential(nn.Linear(d_model*seq_len,3, bias=True))
        self.projection_aiz = nn.Sequential(nn.Linear(d_model * seq_len, 3, bias=True))
        self.BN = nn.BatchNorm1d(d_model*seq_len)
        self.projection_depth = nn.Sequential(nn.Linear(d_model*seq_len, 1, bias=True))
        self.projection_arrivetime = nn.Sequential(nn.Linear(d_model*seq_len, 3, bias=False))
        self.projection_ml = nn.Linear(d_model*seq_len, 1, bias=False)

    def forward(self, x_enc, x_dec, x_dec_lat, x_dec_lng, receive_location, receive_type,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #P波拾取子任务
        enc_out = self.enc_embedding(x_enc)
        #enc_out_arrivetime = self.enc_arrivetime_embedding(x_enc)
        enc_out_arrivetime, attns_arrivetime = self.encoder_arrivetime(enc_out, attn_mask=enc_self_mask)
        enc_out_arrivetime2 = self.attention_2(enc_out_arrivetime)
        dec_out_arrive = self.projection_arrivetime(self.relu(enc_out_arrivetime.view(len(enc_out_arrivetime2), -1)))

        #定位主任务
        #enc_out = self.enc_embedding(x_enc)
        # enc_out_loc = self.start_conv(x_enc)
        # enc_out_loc = self.start_conv2(enc_out_loc)
        # enc_out = self.start_conv3(enc_out_loc)
        #dec_out = self.start_conv4(enc_out_loc)
        enc_out_loc, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #P波到时特征作为注意力添加入主任务网络中
        #enc_out_loc = torch.mul(enc_out_loc, F.normalize(enc_out_arrivetime, dim=1))
        #dec_out_lat = self.dec_embedding_lat(x_dec_lat)
        # dec_out_lat = self.attention_1(enc_out_loc, None)
        # dec_out_lng = self.attention_2(enc_out_loc, None)
        # pro_input_lat = dec_out_lat.view(len(dec_out_lat), dec_out_lat.shape[1] * dec_out_lat.shape[2])
        # pro_input_lng = dec_out_lng.view(len(dec_out_lng), dec_out_lng.shape[1] * dec_out_lng.shape[2])
        #dec_out, attn = self.encoder(enc_out_loc, None)
        enc_out = torch.mul(enc_out_loc, F.softmax(enc_out_arrivetime, dim=1))
        dec_out = (self.decoder(enc_out_loc, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask))
        pro_input = dec_out.view(len(dec_out), dec_out.shape[1] * dec_out.shape[2])
        dec_aiz = self.projection_aiz(pro_input)
        dec_lat = self.projection_lat(((pro_input)))

        #enc_out_depth = self.enc_embedding(x_enc)
        # enc_out_depth = self.start_conv(x_enc)
        # enc_out_depth = self.start_conv2(enc_out_depth)
        # enc_out_depth = self.start_conv3(enc_out_depth)
        # enc_out_depth = self.start_conv4(enc_out_depth)
        enc_out_depth, attns = self.encoder_depth(enc_out, attn_mask=enc_self_mask)
        #dec_out_depth = self.attention_1(enc_out_loc, None)
        # receive_location = receive_location.squeeze()
        # location = torch.cat((receive_location,elevation),1)

        pro_input_depth = enc_out_depth.view(len(enc_out_depth), enc_out_depth.shape[1] * enc_out_depth.shape[2])
        # pro_input_depth = torch.cat((pro_input_depth,location))
        # dec_out_aiz = self.decoder_aiz(enc_out_loc, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # pro_input_aiz = dec_out_aiz.view(len(dec_out), dec_out_aiz.shape[1] * dec_out_aiz.shape[2])
        # enc_out_aiz = self.attention_1(enc_out)
        # pro_input_depth = enc_out_aiz.view(len(enc_out_aiz), enc_out_aiz.shape[1] * enc_out_aiz.shape[2])

        dec_depth = self.projection_depth((pro_input_depth))

        #enc_out_ml = self.enc_embedding_ml(x_enc)
        #enc_out_ml, attns = self.encoder_ml(enc_out_ml, attn_mask=enc_self_mask)
        enc_out_ml = torch.cat((enc_out, dec_out), dim=1)
        #enc_out_ml = torch.cat((enc_out_ml, dec_out), dim=1)
        #enc_out_ml = torch.mul(enc_out_ml, F.normalize(dec_out_lng, dim=1))
        dec_out_ml = self.dec_embedding_ml(x_dec)
        dec_out_ml = self.decoder_ml(dec_out_ml, enc_out_ml, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out_ml = dec_out_ml.view(len(dec_out_ml), dec_out_ml.shape[1] * dec_out_ml.shape[2])
        dec_out_ml = self.projection_ml(dec_out_ml)

        return dec_out_arrive, dec_lat, dec_aiz, dec_depth, dec_out_ml  # [B, L, D]