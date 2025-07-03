from torch import nn


#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    #编码，给一个输出X,输出一个状态
    def forward(self, X, *args):
        raise NotImplementedError
    

#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):#拿到encoder的输出——编码器的输出传递到解码器
        raise NotImplementedError

    def forward(self, X, state):#可以额外输出，不管更新state
        raise NotImplementedError
    

#合并
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)#编码器的输出传递进入解码器的init_state
        return self.decoder(dec_X, dec_state)#解码器的输入和状态传进解码器，得到输出
    
    