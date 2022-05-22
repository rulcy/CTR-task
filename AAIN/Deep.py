import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time
import tensorflow.keras.backend as K

MODEL_DIR = "./checkpoint"
genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess_CTR.p', mode='rb'))
def save_params(params):
    pickle.dump(params, open('params.p', 'wb'))

def load_params():
    return pickle.load(open('params.p', mode='rb'))

embed_dim = 16
#用户ID个数
uid_max = max(features.take(0,1)) + 1 # 6040
#性别个数
gender_max = max(features.take(2,1)) + 1 # 1 + 1 = 2
#年龄类别个数
age_max = max(features.take(3,1)) + 1 # 6 + 1 = 7
#职业个数
job_max = max(features.take(4,1)) + 1 # 20 + 1 = 21
#电影ID个数
movie_id_max = max(features.take(1,1)) + 1 # 3952
#电影类型个数
movie_categories_max = max(genres2int.values()) + 1 # 18 + 1 = 19 ×  26 √
movie_year_max = max(features.take(5,1)) + 1 # 83


# 超参数
num_epochs = 5
batch_size = 1024
dropout_keep = 0.5
learning_rate = 0.001

show_every_n_batches = 20

# 正则化系数
# lambd = 0.001

save_dir = './save'

###################################################################################################


class MultiHeadAttention(keras.Model):
    # 需要重写Call方法来定制模型。
    def __init__(self, model_size, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.model_size = model_size # 模型维度 D
        self.num_heads = num_heads # head数
        self.head_size = model_size // num_heads # 每个head的维数
        self.WQ = keras.layers.Dense(model_size*2)
        self.WK = keras.layers.Dense(model_size*2)
        self.WV = keras.layers.Dense(model_size*2)

    def call(self, query, key, value, mask=None):
        # query: (batch, maxlen, model_size)
        # key  : (batch, maxlen, model_size)
        # value: (batch, maxlen, model_size)
        batch_size = tf.shape(query)[0]

        # shape: (batch, maxlen, model_size)
        # 输出QKV
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)


        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_heads, self.head_size])
            # -1代表现在不确定，根据其他确定这个维度的shape。
            return tf.transpose(x, perm=[0, 2, 1, 3]) # 这里perm意为将第二第三维互换。

        # shape: (batch, num_heads, maxlen, head_size)
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        # shape: (batch, num_heads, maxlen, maxlen)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 缩放 matmul_qk
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            # mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            score += (1 - mask) * -1e9

        alpha = tf.nn.softmax(score)
        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.model_size*2))
        # output = self.dense(context)

        return context

class CrossAttention(keras.Model):
    # 需要重写Call方法来定制模型。
    def __init__(self, model_size, dff,num_heads, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)

        self.model_size = model_size # 模型维度 D
        self.num_heads = num_heads # head数
        self.head_size = model_size // num_heads # 每个head的维数
        self.WQ = keras.layers.Dense(dff)
        self.WK = keras.layers.Dense(dff)
        self.WV = keras.layers.Dense(dff)

    def call(self, query, key, value, mask=None):
        # query: (batch, maxlen, model_size)
        # key  : (batch, maxlen, model_size)
        # value: (batch, maxlen, model_size)
        batch_size = tf.shape(query)[0]
        maxlen = query.shape[1]

        # shape: (batch, maxlen, model_size)
        # 输出QKV
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)


        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_heads, self.head_size])
            # -1代表现在不确定，根据其他确定这个维度的shape。
            return tf.transpose(x, perm=[0, 2, 1, 3]) # 这里perm意为将第二第三维互换。

        # shape: (batch, num_heads, maxlen, head_size)
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        # shape: (batch, num_heads, maxlen, maxlen)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 缩放 matmul_qk
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            # mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            score += (1 - mask) * -1e9

        alpha = tf.nn.softmax(score)
        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, maxlen, self.model_size))
        return context

# Transformer块（自注意力块）
class Block_3_Cross_att(keras.layers.Layer):
    def __init__(self, model_size, num_heads, dff_size, rate=0.1):
        super(Block_3_Cross_att, self).__init__()
        self.attention = MultiHeadAttention(model_size, num_heads)
        self.wres = keras.layers.Dense(model_size*2)

    def call(self, q, k, v, training, mask=None):
        # multi head attention
        attn_output = self.attention(q, k, v, mask)
        wres_out = self.wres(q)
        # residual connection
        out = tf.nn.relu(attn_output+wres_out)
        return out

#CIN ##################################################################################################

def compressed_interaction_net(x0, xl, k, n_filters):
    """
    @param x0: 原始输入
    @param xl: 第l层的输入
    @param k: embedding dim
    @param n_filters: 压缩网络filter的数量
    """
    # 这里设x0中共有m个特征，xl中共有h个特征

    # 1.将x0与xl按照k所在的维度（-1）进行拆分，每个都可以拆成k列
    x0_cols = tf.split(x0, k, axis=-1)  # ?, m, k
    xl_cols = tf.split(xl, k, axis=-1)  # ?, h, k

    assert len(x0_cols) == len(xl_cols), print("error shape!")

    # 2.遍历k列，对于x0与xl所在的第i列进行外积计算，存在feature_maps中
    feature_maps = []
    for i in range(k):
        feature_map = tf.matmul(xl_cols[i], x0_cols[i], transpose_b=True)  # 外积 ?, h, m
        feature_map = tf.expand_dims(feature_map, axis=-1)  # ?, h, m, 1
        feature_maps.append(feature_map)


    # 3.得到 h × m × k 的三维tensor
    feature_maps = tf.keras.layers.Concatenate(axis=-1)(feature_maps)  # ?, h, m, k

    # 3.压缩网络
    x0_n_feats = x0.get_shape()[1]  # m
    xl_n_feats = xl.get_shape()[1]  # h
    reshaped_feature_maps = tf.keras.layers.Reshape(target_shape=(x0_n_feats * xl_n_feats, k))(feature_maps)  # ?, h*m, k
    transposed_feature_maps = tf.transpose(reshaped_feature_maps, [0, 2, 1])  # ?, k, h*m

    new_feature_maps = tf.keras.layers.Conv1D(n_filters, 1, 1)(transposed_feature_maps)  # ?, k, n_filters
    new_feature_maps = tf.transpose(new_feature_maps, [0, 2, 1])  # ?, n_filters, k

    return new_feature_maps

def build_cin(x0, k=16, n_layers=3, n_filters=20):
    """
    构建多层CIN网络
    @param x0: 原始输入的feature maps: ?, m, k
    @param k: 特征embedding的维度
    @param n_layers: CIN网络层数
    @param n_filters: 每层CIN网络输出的feature_maps的个数
    164661个参数。
    """
    # cin layers
    cin_layers = []
    # 存储每一层cin sum pooling的结果
    pooling_layers = []
    xl = x0
    for layer in range(n_layers):
        xl = compressed_interaction_net(x0, xl, k, n_filters)
        cin_layers.append(xl)
        # sum pooling
        pooling = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=-1))(xl)
        pooling_layers.append(pooling)

    # 将所有层的pooling结果concat起来
    output = tf.keras.layers.Concatenate(axis=-1)(pooling_layers)

    return output

def build_att_cin(x0, k=16, n_layers=4, n_filters=12):
    """
    构建多层CIN网络
    @param x0: 原始输入的feature maps: ?, m, k
    @param k: 特征embedding的维度
    @param n_layers: CIN网络层数
    @param n_filters: 每层CIN网络输出的feature_maps的个数
    168549个参数.
    多的参数量来自于x0不同和attention_net。
    """
    # cin layers
    cin_layers = []
    # 存储每一层cin sum pooling的结果
    pooling_layers = []
    # xl = x0


    #####################2要定义多个，不是一个。
    Attention_Block = CrossAttention(model_size=embed_dim, dff=embed_dim, num_heads=2)
    x0_att = Attention_Block(x0, x0, x0)
    xl = x0_att
    for layer in range(n_layers):
        xl = compressed_interaction_net(x0_att, xl, k, n_filters)
        cin_layers.append(xl)
        # sum pooling
        pooling = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=-1))(xl)
        pooling_layers.append(pooling)
        # Creat Attention
        if layer<n_layers-1:
            x0_att = CrossAttention(model_size=embed_dim, dff=embed_dim, num_heads=2)(xl, x0, x0)

    # 将所有层的pooling结果concat起来
    output = tf.keras.layers.Concatenate(axis=-1)(pooling_layers)

    return output

###################################################################################################
def get_inputs():
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')
    user_gender = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_gender')
    user_age = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_age')
    user_job = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_job')

    movie_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_id')
    movie_categories = tf.keras.layers.Input(shape=(18,), dtype='int32', name='movie_categories')
    movie_year = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_year')
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_year

def get_user_embedding(uid, user_gender, user_age, user_job):
    # tf.keras.layers.Embedding(字典大小，输出维度，输入数据最大维度)
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1, name='uid_embed_layer')(uid)
    gender_embed_layer = tf.keras.layers.Embedding(gender_max, embed_dim, input_length=1, name='gender_embed_layer')(user_gender)
    age_embed_layer = tf.keras.layers.Embedding(age_max, embed_dim, input_length=1, name='age_embed_layer')(user_age)
    job_embed_layer = tf.keras.layers.Embedding(job_max, embed_dim, input_length=1, name='job_embed_layer')(user_job)
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

def get_movie_embedding(movie_id, movie_categories,movie_year):
    movie_id_embed_layer = tf.keras.layers.Embedding(movie_id_max, embed_dim, input_length=1, name='mvid_embed_layer')(movie_id)
    movie_categories_embed_layer = tf.keras.layers.Embedding(movie_categories_max, embed_dim, input_length=18, name='mv_categories_embed_layer')(movie_categories)
    movie_year_embed_layer = tf.keras.layers.Embedding(movie_year_max, embed_dim, input_length=1, name='mv_year_embed_layer')(movie_year)
    # 类别型数据平铺
    movie_categories_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        movie_categories_embed_layer)
    return movie_id_embed_layer, movie_categories_embed_layer, movie_year_embed_layer

##############################################################################################################################

def get_DeepCrossing(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer,movie_id_embed_layer, movie_categories_embed_layer, movie_year_embed_layer):
    embs = tf.keras.layers.concatenate(
        [uid_embed_layer,gender_embed_layer, age_embed_layer, job_embed_layer,movie_id_embed_layer, movie_categories_embed_layer,
         movie_year_embed_layer],
        1)
    layer_for_concat = []
    for i in range(embs.shape[1]):
        layer_for_concat.append(tf.keras.layers.Dense(embed_dim,name=f'concat_dense{i}',activation='relu')(embs[:,i:i+1,:]))
    embs_concat = tf.keras.layers.concatenate(layer_for_concat,2)

    # 第二、三层全连接
    fc2_layer = tf.keras.layers.Dense(100, activation='relu')(embs_concat)  # (?, 1, 100)
    fc3_layer = tf.keras.layers.Dense(100, activation='relu')(fc2_layer)  # (?, 1, 100)
    fc4_layer = tf.keras.layers.Dense(100, activation='relu')(fc3_layer)  # (?, 1, 100)
    fc4_layer_flat = tf.keras.layers.Reshape([100], name="fc5_layer_flat")(fc4_layer)
    return fc4_layer_flat

class mv_network(object):
    def __init__(self, batch_size=batch_size):
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}
        self.best_metrics = 0

        # 获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_year = get_inputs()
        # 获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                                   user_age, user_job)
        # 获取Movie的3个嵌入向量
        movie_id_embed_layer, movie_categories_embed_layer, movie_year_embed_layer = get_movie_embedding(movie_id,
                                                                                                         movie_categories,
                                                                                                         movie_year)
        # 获取主体模型
        Deep_layer_flat = get_DeepCrossing(uid_embed_layer, gender_embed_layer, age_embed_layer,
                                                                job_embed_layer, movie_id_embed_layer,
                                                                movie_categories_embed_layer, movie_year_embed_layer)

        inference = tf.keras.layers.Dense(1, name="inference", activation='sigmoid')(Deep_layer_flat)  # inference_dense


        # 选择输入输出
        self.model = tf.keras.Model(
            inputs=[uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_year],
            outputs=[inference])

        # 输出各层模型的参数状况
        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # MSE损失，将计算值回归到评分
        self.ComputeLoss = tf.keras.losses.BinaryCrossentropy()
        # self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.AUC()

        # 创建文件夹
        if tf.io.gfile.exists(MODEL_DIR):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        # 保存模型
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints-1')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # 重新加载保存的模型
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    @tf.function
    def train_step(self, x, y):
        w = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(w)
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6]
                                 ], training=True)
            loss = self.ComputeLoss(y, logits)
            self.ComputeMetrics(y, logits)
        # 计算梯度。
        grads = tape.gradient(loss, self.model.trainable_variables)
        # 应用梯度。
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits # 损失；结果。

    def training(self, features, targets_values, epochs=5, log_freq=50):
        overfit_bool = 0 # 判断过拟合
        trains_X, eval_X, trains_y, eval_y = train_test_split(features,
                                                            targets_values,
                                                            test_size=0.1)
        train_X, test_X, train_y, test_y = train_test_split(trains_X,
                                                            trains_y,
                                                            test_size=0.1)
        for epoch_i in range(epochs):

            train_batches = get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            if True:
                start = time.time()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    categories = np.zeros([self.batch_size, 18])
                    for i in range(self.batch_size):
                        categories[i] = x.take(6, 1)[i]

                    loss, logits = self.train_step([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                                    categories.astype(np.float32),
                                                    np.reshape(x.take(5, 1), [self.batch_size, 1]).astype(np.float32)
                                                    ],
                                                   np.reshape(y, [self.batch_size, 1]).astype(np.float32))

                    avg_loss(loss)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} AUC: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result()), rate))

                        # 重置值。
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        start = time.time()
            train_end = time.time()
            print(
                '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                         train_end - train_start))

            self.evaling((eval_X,eval_y))

            if self.testing((test_X, test_y)):
                overfit_bool += 1
            else:
                overfit_bool = 0
            if overfit_bool == 2:
                break

    def testing(self, test_dataset):
        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)

        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.ComputeMetrics.reset_states()

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, 18])
            for i in range(self.batch_size):
                categories[i] = x.take(6, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                 categories.astype(np.float32),
                                 np.reshape(x.take(5, 1), [self.batch_size, 1]).astype(np.float32)
                                 ],
                                 training=False)
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)

            # 保存测试损失
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)

        Metrics_temp = self.ComputeMetrics.result()
        print('Model test set loss: {:0.6f} AUC: {:0.6f}'.format(avg_loss.result(), Metrics_temp))

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
        else:
            print("best loss = {}".format(self.best_loss))

        if Metrics_temp  < self.best_metrics:
            print("best metrics = {}".format(self.best_metrics))
            self.ComputeMetrics.reset_states()
            return 1
        else:
            self.best_metrics = Metrics_temp
            print("best metrics = {}".format(self.best_metrics))
            self.ComputeMetrics.reset_states()
            self.checkpoint.save(self.checkpoint_prefix)
            return 0

    def evaling(self,eval_data):
        eval_X, eval_y = eval_data
        test_batches = get_batches(eval_X, eval_y, self.batch_size)

        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.ComputeMetrics.reset_states()

        batch_num = (len(eval_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, 18])
            for i in range(self.batch_size):
                categories[i] = x.take(6, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                 categories.astype(np.float32),
                                 np.reshape(x.take(5, 1), [self.batch_size, 1]).astype(np.float32)
                                 ],
                                training=False)

            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)

            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)

        Metrics_temp = self.ComputeMetrics.result()
        print('Model Eval set loss: {:0.6f} AUC: {:0.6f}'.format(avg_loss.result(), Metrics_temp))

    def forward(self, xs):
        predictions = self.model(xs)
        return predictions

def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

if __name__ == '__main__':
    mv_net = mv_network()
    mv_net.training(features, targets_values, epochs=10)
    pass