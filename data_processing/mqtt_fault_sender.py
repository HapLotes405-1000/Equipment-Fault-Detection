import pandas as pd
import json
import time
import paho.mqtt.client as mqtt
import numpy as np

# ===================== 1. 配置项（需根据算法端实际信息修改） =====================
# MQTT连接配置（算法端主机）
MQTT_CONFIG = {
    "broker": "192.168.1.100",  # 算法端MQTT服务器IP/域名
    "port": 1883,  # MQTT端口，默认1883
    "topic": "bearing/fault/detection",  # 算法端订阅的主题（需提前约定）
    "username": "algorithm_user",  # 算法端MQTT认证用户名（无则设为None）
    "password": "algorithm_pwd",  # 算法端MQTT认证密码（无则设为None）
    "qos": 1  # QoS级别，1=至少送达一次（故障数据不丢失）
}

# 数据文件路径
DATA_CONFIG = {
    "train_data_path": "D:\\X program\\training_data\\train.csv",  # 训练集路径（有label）
    "test_data_path": "D:\\X program\\training_data\\test_data.csv"  # 测试集路径（无label）
}

# 轴承故障编码映射（标准化定义，与算法端对齐）
FAULT_CODE_MAP = {
    0: {"code": "B000", "desc": "直径1-正常状态", "level": "normal"},
    1: {"code": "B101", "desc": "直径1-内圈故障", "level": "fault"},
    2: {"code": "B102", "desc": "直径1-外圈故障", "level": "fault"},
    3: {"code": "B103", "desc": "直径1-滚珠故障", "level": "fault"},
    4: {"code": "B201", "desc": "直径2-内圈故障", "level": "fault"},
    5: {"code": "B202", "desc": "直径2-外圈故障", "level": "fault"},
    6: {"code": "B203", "desc": "直径2-滚珠故障", "level": "fault"},
    7: {"code": "B301", "desc": "直径3-内圈故障", "level": "fault"},
    8: {"code": "B302", "desc": "直径3-外圈故障", "level": "fault"},
    9: {"code": "B303", "desc": "直径3-滚珠故障", "level": "fault"}
}


# ===================== 2. 数据读取与预处理函数 =====================
def load_bearing_data(file_path, is_train=True):
    """
    读取轴承故障数据集，预处理振动信号数据
    :param file_path: 数据文件路径
    :param is_train: 是否为训练集（训练集含label，测试集不含）
    :return: 预处理后的数据集（DataFrame）
    """
    try:
        # 读取CSV文件（6000维振动信号+id+label（训练集））
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f" 成功读取数据：{file_path}，共{len(df)}条样本，{len(df.columns)}列")

        # 提取振动信号列（1~6000列）
        signal_cols = [col for col in df.columns if str(col).isdigit() and 1 <= int(col) <= 6000]
        # 计算振动信号关键特征（轻量化，避免6000维数据直接传输）
        df["signal_mean"] = df[signal_cols].mean(axis=1)  # 信号均值
        df["signal_std"] = df[signal_cols].std(axis=1)  # 信号标准差（反映波动）
        df["signal_max"] = df[signal_cols].max(axis=1)  # 信号最大值
        df["signal_min"] = df[signal_cols].min(axis=1)  # 信号最小值
        df["signal_peak_to_peak"] = df["signal_max"] - df["signal_min"]  # 峰峰值（故障核心特征）

        # 训练集补充故障编码信息
        if is_train and "label" in df.columns:
            df["fault_code"] = df["label"].map(lambda x: FAULT_CODE_MAP.get(x, {}).get("code", "UNKNOWN"))
            df["fault_desc"] = df["label"].map(lambda x: FAULT_CODE_MAP.get(x, {}).get("desc", "未知状态"))
            df["fault_level"] = df["label"].map(lambda x: FAULT_CODE_MAP.get(x, {}).get("level", "unknown"))
        else:
            # 测试集无label，标记为待预测
            df["fault_code"] = "PENDING"
            df["fault_desc"] = "待算法端预测故障状态"
            df["fault_level"] = "pending"

        return df

    except Exception as e:
        print(f" 读取数据失败：{str(e)}")
        return pd.DataFrame()


# ===================== 3. MQTT连接与消息发送函数 =====================
def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT连接回调函数"""
    rc_msg = {
        0: "连接成功",
        1: "协议版本错误",
        2: "客户端ID非法",
        3: "服务器不可用",
        4: "用户名/密码错误",
        5: "未授权"
    }
    if rc == 0:
        print(f" MQTT连接成功（算法端主机：{MQTT_CONFIG['broker']}）")
    else:
        print(f" MQTT连接失败：{rc_msg.get(rc, f'未知错误码{rc}')}")


def on_mqtt_publish(client, userdata, mid):
    """MQTT消息发布回调函数"""
    print(f" 消息ID {mid} 已发送至算法端")


def send_bearing_data_to_algorithm(df):
    """
    将轴承数据封装为MQTT消息发送给算法端
    :param df: 预处理后的轴承数据集
    """
    # 1. 创建MQTT客户端
    client = mqtt.Client(client_id=f"bearing_sender_{int(time.time())}",
                         callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_mqtt_connect
    client.on_publish = on_mqtt_publish

    # 2. 设置认证（若无则跳过）
    if MQTT_CONFIG["username"] and MQTT_CONFIG["password"]:
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])

    # 3. 连接算法端MQTT服务器
    try:
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=60)
        client.loop_start()  # 启动后台循环
        time.sleep(1)  # 等待连接稳定

        # 4. 逐条封装并发送消息
        send_count = 0
        fail_count = 0
        for _, row in df.iterrows():
            # 封装MQTT消息体（轻量化，仅含算法端需的核心特征）
            mqtt_msg = {
                "sample_id": int(row["id"]),  # 样本唯一ID
                "fault_code": row["fault_code"],  # 故障编码（训练集有/测试集待预测）
                "fault_desc": row["fault_desc"],  # 故障描述
                "fault_level": row["fault_level"],  # 故障级别（normal/fault/pending）
                # 振动信号关键特征（算法端用于故障检测）
                "signal_features": {
                    "mean": round(float(row["signal_mean"]), 6),
                    "std": round(float(row["signal_std"]), 6),
                    "max": round(float(row["signal_max"]), 6),
                    "min": round(float(row["signal_min"]), 6),
                    "peak_to_peak": round(float(row["signal_peak_to_peak"]), 6)
                },
                "send_timestamp": int(time.time()),  # 发送时间戳（秒级）
                "send_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 发送时间
            }

            # 转换为JSON字符串（支持中文，避免乱码）
            msg_json = json.dumps(mqtt_msg, ensure_ascii=False)

            # 发布消息到算法端主题
            result = client.publish(
                topic=MQTT_CONFIG["topic"],
                payload=msg_json,
                qos=MQTT_CONFIG["qos"]
            )

            # 统计发送结果
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                send_count += 1
                print(f" 样本{row['id']}发送成功 | 故障编码：{row['fault_code']}")
            else:
                fail_count += 1
                print(f" 样本{row['id']}发送失败 | 错误码：{result.rc}")

            # 控制发送速率（避免算法端处理不过来）
            time.sleep(0.2)

        # 5. 断开连接
        time.sleep(2)
        client.loop_stop()
        client.disconnect()

        # 打印发送统计
        print("\n===== MQTT发送统计 =====")
        print(f"总样本数：{len(df)}")
        print(f"发送成功：{send_count}")
        print(f"发送失败：{fail_count}")
        print(f"成功率：{round(send_count / len(df) * 100, 2)}%")

    except Exception as e:
        print(f" MQTT发送异常：{str(e)}")


# ===================== 4. 主执行流程 =====================
if __name__ == "__main__":
    # 选择读取训练集/测试集（按需切换）
    # ---------- 读取训练集（含故障标签） ----------
    train_df = load_bearing_data(DATA_CONFIG["train_data_path"], is_train=True)
    if not train_df.empty:
        print("\n 开始向算法端发送训练集故障数据...")
        send_bearing_data_to_algorithm(train_df)

    # ---------- 读取测试集（无故障标签，待预测） ----------
    # test_df = load_bearing_data(DATA_CONFIG["test_data_path"], is_train=False)
    # if not test_df.empty:
    #     print("\n开始向算法端发送测试集待预测数据...")
    #     send_bearing_data_to_algorithm(test_df)