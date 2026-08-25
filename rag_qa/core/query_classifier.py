#查询分类器,基于bert模型实现查询分类，将用户查询分为通用知识和业务咨询两类，便于后续控制是否回答与业务无关的问题
#使用业务数据对模型微调，提高对问题意图识别的表现

import json
import os
import torch
import sys
from base.logger import logger
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(os.path.abspath(current_dir)))
project_root = os.path.abspath(os.path.dirname(os.path.abspath(rag_qa_path)))
sys.path.insert(0, project_root)


class QueryClassifier:
    def __init__(self, model_path='../models/bert_query_classifier'):
       # 加载bert
        self.pre_trained_model_path = f'{rag_qa_path}/models/bert-base-chinese'
        # 模型训练以后保存的位置
        self.model_path = model_path
        # 加载 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu" )
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            # 加载预训练模型并转移到指定设备
            logger.info("初始化新 BERT 模型")
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 初始化新模型
            logger.info("初始化预训练 BERT 模型")
            self.model = BertForSequenceClassification.from_pretrained(self.pre_trained_model_path, num_labels=2)
            self.model.to(self.device)


    def save_model(self):
        """
        保存模型
        """
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"模型已保存至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        """
        预处理数据为 BERT 输入格式
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        """
        创建 PyTorch 数据集
        """

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file):
        """
        训练 BERT 分类模型
        """
        # 加载数据集
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # 数据划分
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # 创建数据集
        train_dataset = self.create_dataset(train_encodings, train_labels)
        # print(f'train_dataset--》{train_dataset[0]}')
        val_dataset = self.create_dataset(val_encodings, val_labels)
        #
        # 设置训练参数
        training_args = TrainingArguments(
            # 设置模型和检查点保存的目录路径
            output_dir="./bert_results",
            # 设置训练的总轮数为3轮
            num_train_epochs=3,
            # 设置每个设备（GPU/CPU）上的训练批次大小为8
            per_device_train_batch_size=8,
            # 设置每个设备（GPU/CPU）上的评估批次大小为8
            per_device_eval_batch_size=8,
            # 设置学习率预热步数为500步，训练初期学习率从0逐渐增加到设定值
            warmup_steps=500,
            # 设置权重衰减系数为0.01，用于防止过拟合
            weight_decay=0.01,
            # 设置日志文件保存的目录路径
            logging_dir="./bert_logs",
            # 设置每10个训练步骤记录一次日志
            logging_steps=10,
            # 设置评估策略为每个epoch结束后进行评估
            evaluation_strategy="epoch",
            # 设置模型保存策略为每个epoch结束后保存
            save_strategy="epoch",
            # 设置训练结束后加载最佳模型而非最后一个模型
            load_best_model_at_end=True,
            # 设置最多保存1个检查点文件，超出时自动删除旧的
            save_total_limit=1,
            # 设置用于判断最佳模型的指标为评估损失
            metric_for_best_model="eval_loss",
            # 禁用FP16混合精度训练，使用FP32精度
            fp16=False,
        )

        # 初始化 Trainer
        trainer = Trainer(
            # 传入要训练的模型实例
            model=self.model,
            # 传入上面定义的训练参数配置
            args=training_args,
            # 传入训练数据集
            train_dataset=train_dataset,
            # 传入验证数据集，用于训练过程中评估模型性能
            eval_dataset=val_dataset,
            # 传入计算评估指标的函数，用于在验证集上计算准确率等指标
            compute_metrics=self.compute_metrics
        )
        # 训练模型
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()

        # 评估模型
        self.evaluate_model(val_texts, val_labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """评估模型性能"""
        # 仅对 texts 进行分词，labels 已为数字
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels  # 直接使用数字标签

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    def predict_category(self, query):
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"

if __name__ == "__main__":
    # 训练模型,将训练用的数据放置于对应目录下，train_model的传参为训练数据文件对query_classifier.py文件的相对路径。
    # 训练参数见126-155行
    # 运行此测试方法即可手动进行预训练模型的微调，使用自己的业务数据进行效果更佳。
    ## classifier.train_model("../classify_data/model_generic_5000.json")
    pass
