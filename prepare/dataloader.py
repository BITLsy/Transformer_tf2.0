import os
import tensorflow as tf
import tensorflow_datasets as tfds

from cfg import *

BUFFER_SIZE = 20000


def load_examples():
    """ 加载数据集pt_to_en，自动下载到 "~/tensorflow_datasets/" 路径下 """
    examples, metadata = tfds.load(CONFIG['data']['origin'], with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    print('Loaded examples')
    return train_examples, val_examples
    

def build_tokenizer(examples):
    """ 构建tokenizer """
    en_vocab_path_prefix = get_path(CONFIG['vocab']['en_vocab_path_prefix'])
    pt_vocab_path_prefix = get_path(CONFIG['vocab']['pt_vocab_path_prefix'])
    
    if not os.path.isdir(get_path(CONFIG['vocab']['dir'])):
        os.makedirs(get_path(CONFIG['vocab']['dir']))
        
    # 加载en的tokenizer
    if not os.path.exists(en_vocab_path_prefix + '.subwords'):
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in examples), target_vocab_size=2**13)
        tokenizer_en.save_to_file(filename_prefix=en_vocab_path_prefix)
        print('Build tokenizer for en')
    else:
        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix=en_vocab_path_prefix)
        print('Load tokenizer for en')

    # 加载pt的tokenizer
    if not os.path.exists(pt_vocab_path_prefix + '.subwords'):
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in examples), target_vocab_size=2 ** 13)
        tokenizer_pt.save_to_file(filename_prefix=pt_vocab_path_prefix)
        print('Build tokenizer for pt ')
    else:
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix=pt_vocab_path_prefix)
        print('Load tokenizer for pt')

    return tokenizer_en, tokenizer_pt


def gen_dataset(batch_size=64):
    """ 生成训练和验证batched数据集 """
    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
        
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2
    
    def filter_max_length(x, y, max_length=40):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)
    
    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

    train_examples, val_examples = load_examples()
    tokenizer_en, tokenizer_pt = build_tokenizer(train_examples)
    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    
    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(batch_size, ([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(batch_size, ([-1], [-1]))
    
    return train_dataset, val_dataset, input_vocab_size, target_vocab_size


if __name__ == '__main__':
    
    sample_string = 'Transformer is awesome.'
    
    # tokenized_string = tokenizer_en.encode(sample_string)
    # print('Tokenized string is {}'.format(tokenized_string))
    #
    # original_string = tokenizer_en.decode(tokenized_string)
    # print('The original string: {}'.format(original_string))
    #
    # assert original_string == sample_string

    # _, val_dataset = gen_dataset()
    # pt_batch, en_batch = next(iter(val_dataset))
    # pt_batch, en_batch

