import numpy as np

# # 定义卡尔曼滤波器参数
# dt = 0.1  # 时间间隔
# A = np.array([[1, dt], [0, 1]])  # 状态转移矩阵
# H = np.array([[1, 0], [0, 1]])  # 测量矩阵
# Q = np.array([[0.1, 0], [0, 0.1]])  # 过程噪声协方差矩阵
# R = np.array([[1, 0], [0, 1]])  # 测量噪声协方差矩阵
# x = np.array([[0], [0]])  # 初始状态向量
# P = np.array([[1, 0], [0, 1]])  # 初始状态协方差矩阵

# # 定义测量值
# measurements = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# # 定义预测结果数组
# predictions = []

# # 开始卡尔曼滤波预测
# for measurement in measurements:
#     # 预测步骤
#     measurement = np.expand_dims(measurement, axis=1)
#     x = np.dot(A, x)
#     P = np.dot(np.dot(A, P), A.T) + Q

#     # 更新步骤
#     K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
#     x = x + np.dot(K, (measurement - np.dot(H, x)))
#     P = np.dot((np.eye(2) - np.dot(K, H)), P)

#     # 保存预测结果
#     print(x.shape)
#     print(P.shape)
#     predictions.append(x)

# # 打印预测结果
# print(predictions)
# # print(predictions[0].shape)


class kalman_simple(object):
    def __init__(self,dt=0.1):
        
        self.dt = dt  # 时间间隔
        self.A = np.array([[1, self.dt], [0, 1]])  # 状态转移矩阵
        self.H = np.array([[1, 0], [0, 1]])  # 测量矩阵
        self.Q = np.array([[0.1, 0], [0, 0.1]])  # 过程噪声协方差矩阵
        self.R = np.array([[1, 0], [0, 1]])  # 测量噪声协方差矩阵
        self.x = np.array([[0], [0]])  # 初始状态向量
        self.P = np.array([[1, 0], [0, 1]])  # 初始状态协方差矩阵
        
    def predict(self,measurement):
        measurement = np.expand_dims(measurement, axis=1)
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # 更新步骤
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        self.x = self.x + np.dot(self.K, (measurement - np.dot(self.H, self.x)))
        self.P = np.dot((np.eye(2) - np.dot(self.K, self.H)), self.P)
        
        return self.x
    
    
if __name__ == '__main__':
    print("Kalman filter")
    pass