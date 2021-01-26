---
typora-root-url: ..\img_pytorch
---

## [加速 PyTorch 模型训练的 9 个技巧](https://mp.weixin.qq.com/s/mHm8b_tftwvGDmG91KQoxA)

1. 使用DataLoaders
2. DataLoaders中的workers数量
3. Batch Size
4. 梯度累计（模拟Batch Size）
5. 保留的计算图
6. 移动到单个GPU
7. 16-bit混合精读训练
8. 移动到多个GPUs中（模型复制）
9. 移动到多个GPU-nodes中
10. 思考模型加速的技巧



## [核心开发者全面解读PyTorch内部机制](https://mp.weixin.qq.com/s/jEBn1__kt4njJR28Uhi_rw)   [英文](http://blog.ezyang.com/2019/05/pytorch-internals/)

<p align="center">
	<img width=80% src="Tensorwebp.webp"/>  
</p>



## [Horovod](https://github.com/horovod/horovod)

```python
import tensorflow as tf
import horovod.tensorflow as hvd


# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

