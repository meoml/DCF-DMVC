基于GCN和注意力机制的多视图聚类
1、环境：
torch
numpy
pandas
scikit-learn

2、代码结构说明：
run_tuning.py：训练主程序，包含数据集加载与参数配置
model.py：模型代码
loss.py：损失函数
metric.py：评估指标
run.sh：启动训练的shell脚本（后台挂起+日志输出）

3、快速运行方式：
sh run.sh