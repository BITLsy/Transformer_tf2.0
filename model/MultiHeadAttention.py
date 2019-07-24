import tensorflow as tf
from tensorflow import keras


def scaled_dot_product_attention(q, k, v, mask):
    """
    dot_product attention
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_kv, depth)
        v: value shape == (..., seq_len_kv, depth_v)
    :return: output shape == (..., seq_len_q, depth_v)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 转换数据类型
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logit = matmul_qk / tf.math.sqrt(dk)    # (..., seq_len_q, seq_len_kv)
    
    if mask is not None:
        # mask的位置为1，加上负无穷大，对 k 超过原始序列的位置mask
        scaled_attention_logit += (mask * -1e9)              # (..., seq_len_q, seq_len_kv(masked))

    # (..., seq_len_q, seq_len_kv)
    attention_weights = tf.nn.softmax(scaled_attention_logit, axis=-1)
    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        
        self.dense = keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)     # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)     # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(v, batch_size)     # (batch_size, num_heads, seq_len_q, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, d_model)
        scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(scaled_attention)
        
        return output, attention_weights
    
    
if __name__ == '__main__':
    ####################
    #   测试 Attention
    ####################
    def print_out(q, k, v, mask):
        temp_out, temp_attn = scaled_dot_product_attention(
            q, k, v, mask)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)


    temp_k = tf.constant([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 0, 1]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    temp_q = tf.constant([[0, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=tf.float32)
    
    mask = tf.constant([[0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [1, 1, 1, 1]
                        ], dtype=tf.float32)
    
    print_out(temp_q, temp_k, temp_v, None)
    
    ####################
    #   测试 MultiHA
    ####################
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))
    out, attn = mha(y, k=y, v=y, mask=None)
    print(out.shape)
    print(attn.shape)
