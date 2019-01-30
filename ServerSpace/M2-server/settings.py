# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_HEIGHT = 444
IMAGE_WIDTH = 592
IMAGE_CHANS = 3
DATA_TYPE = "float32"

# 三代机共用异常模型
ModelPath_single_abnormal = "savemodel/V3_single-abnormal_model_v8_sn_20181210.h5"

# V3 model path 三代机矮层
# ModelPath_single_count = "savemodel/V3_low-height_single_count_inception_resnet_v2_0.9593_wudi.h5"

# V3 model path 三代机高层
ModelPath_single_count = "savemodel/V3_high-height_single-count_model_v9lc_0.9926_wudi.h5"


# column
COLUMN_NUM = 7

# initialize constants used for server queuing
GPU_COUNT = 2
QUEUE_GPUS_R1 = ["gpu" + str(i) + "_r1" for i in range(GPU_COUNT)]
QUEUE_GPUS_R2 = ["gpu" + str(i) + "_r2" for i in range(GPU_COUNT)]
QUEUE_GPUS_R3 = ["gpu" + str(i) + "_r3" for i in range(GPU_COUNT)]

# batch size and sleep time
BATCH_SIZE = 5
SERVER_SLEEP = 0.15
CLIENT_SLEEP = 0.15
GET_MAX = 100

# threshold of model
Threshold_Single_Count = 0.65
Threshold_Single_Abnormal = 0.75
Threshold_Double_Abnormal = 0.9
