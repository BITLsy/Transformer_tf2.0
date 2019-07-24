import tensorflow as tf
from tensorflow import keras

from model.PositionalEncoding import positional_encoding
from model.MultiHeadAttention import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
    """返回一个模型， Sequence是一个模型，有call方法"""
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(d_model)
    ])


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
        
        
class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # 自注意力 (只能看到预测位置及左侧的输入)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # 注意力查看encoder输出 (mask掉超过原始长度的部分)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

 
class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        return x
        

class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
            
        return x, attention_weights
        
    
class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)
        self.final_layer = keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        
        dec_output, attention_weights = self.decoder(tar, enc_output, training,
                                                     look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights


if __name__ == '__main__':
    # ************************************
    # 测试position encoder
    # ************************************
    # pos_encoding = positional_encoding(50, 512)
    # print(pos_encoding.shape)
    #
    # plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    # plt.xlabel('Depth')
    # plt.xlim((0, 512))
    # plt.ylabel('Position')
    # plt.colorbar()
    # plt.show()
    
    # ************************************
    # 测试 point_wise_feed_forward_network
    # ************************************
    # sample_ffn = point_wise_feed_forward_network(512, 2048)
    # sample_ffn(tf.random.uniform((64, 50, 512))).shape

    # ************************************
    # 测试 EncoderLayer
    # ************************************
    # sample_encoder_layer = EncoderLayer(512, 8, 2048)
    # sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 50, 512)), False, None)
    # sample_encoder_layer_output.shape
    #
    # sample_decoder_layer = DecoderLayer(512, 8, 2048)
    #
    # sample_decoder_layer_output, _, _ = sample_decoder_layer(
    #     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
    #     False, None, None)
    #
    # sample_decoder_layer_output.shape

    # ************************************
    # 测试 Encoder
    # ************************************
    # sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
    #                          dff=2048, input_vocab_size=8500)
    # sample_encoder_output = sample_encoder(tf.random.uniform([32, 64]),
    #                                        training=False, mask=None)
    # print(sample_encoder_output.shape)
    #
    # # ************************************
    # # 测试 Decoder
    # # ************************************
    # sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
    #                          dff=2048, target_vocab_size=8000)
    #
    # output, attn = sample_decoder(tf.random.uniform((64, 26)),
    #                               enc_output=sample_encoder_output,
    #                               training=False, look_ahead_mask=None,
    #                               padding_mask=None)

    # ************************************
    # 测试 Transformer
    # ************************************
    # sample_transformer = Transformer(
    #     num_layers=2, d_model=512, num_heads=8, dff=2048,
    #     input_vocab_size=8500, target_vocab_size=8000)
    #
    # temp_input = tf.random.uniform((64, 62))
    # temp_target = tf.random.uniform((64, 26))
    #
    # fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
    #                                enc_padding_mask=None,
    #                                look_ahead_mask=None,
    #                                dec_padding_mask=None)
    #
    # fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
    pass


