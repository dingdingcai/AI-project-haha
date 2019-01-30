# import normal packages
import os
import json
import time
import redis
import numpy as np
import pickle
# import my packages
import settings
from DepthwiseConv2D import DepthwiseConv2D
from switchnorm import SwitchNormalization
# choose one of the gpus
with open("gpu_setting.json", 'r') as load_f:
    load_dict = json.load(load_f)
    gpu_index = load_dict["gpu_now"]
    if gpu_index >= settings.GPU_COUNT - 1:
        load_dict["gpu_now"] = 0
    else:
        load_dict["gpu_now"] = gpu_index + 1
with open("gpu_setting.json", 'w') as dump_f:
    json.dump(load_dict, dump_f)
# gpu visible setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
# keras
from keras.models import load_model
# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
# other
image_shape = (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS)
data_type = settings.DATA_TYPE

def predict_process():
    # load models
    print("* Loading models...")
    model_single_abnormal = load_model(settings.ModelPath_single_abnormal,
                                       custom_objects={'SwitchNormalization': SwitchNormalization}, compile=False)
    print("1 single_abnormal model loaded from {}".format(settings.ModelPath_single_abnormal))
    # Temporally delete double_abnormal
    # model_double_abnormal = load_model(settings.ModelPath_double_abnormal,
    #                                    custom_objects={'DepthwiseConv2D': DepthwiseConv2D}, compile=False)
    # print("2 double_abnormal model loaded from {}".format(settings.ModelPath_double_abnormal))
    model_single_count = load_model(settings.ModelPath_single_count,
                                    custom_objects={'DepthwiseConv2D': DepthwiseConv2D}, compile=False)
    print("3 single_count model loaded from {}".format(settings.ModelPath_single_count))
    # continually pool for new request
    while True:
        # time
        loop_start_time = time.time()
        # Queue of take_goods
        queue_r1 = db.lrange(settings.QUEUE_GPUS_R1[gpu_index], 0, settings.BATCH_SIZE - 1)
        batch_size = len(queue_r1)
        # check to see if the batch list is None
        if batch_size > 0:
            # time
            start_time = time.time()
            requestIDs = []
            batch_a = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            batch_b = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            # Temporally delete double_abnormal
            # batch_c = np.zeros(((batch_size,) + (360, 480, 3)), dtype=data_type)
            # batch_b2 = np.zeros(((batch_size,) + (360, 480, 3)), dtype=data_type)
            batch_lc = np.zeros((batch_size, 2 * settings.COLUMN_NUM), dtype=data_type)
            # loop over the queue
            for i, q in enumerate(queue_r1):
                q = pickle.loads(q)
                # deserialize the object and obtain the input image
                image_a = db.get(q["image_a_key"])
                if image_a is not None:
                    batch_a[i] = pickle.loads(image_a)
                image_b = db.get(q["image_b_key"])
                if image_b is not None:
                    batch_b[i] = pickle.loads(image_b)
                # Temporally delete double_abnormal
                # image_c = db.get(q["image_c_key"])
                # if image_c is not None:
                #     batch_c[i] = pickle.loads(image_c)
                # image_b2 = db.get(q["image_b2_key"])
                # if image_b2 is not None:
                #     batch_b2[i] = pickle.loads(image_b2)
                lc = np.array(json.loads(q["column_lc"]))
                batch_lc[i] = lc
                # update the list of image IDs
                requestIDs.append(q["request_id_diy"])
            # do things of single_abnormal detect model
            time_0 = time.time()
            # normal = 0, abnormal = 1
            preds_single_abnormal = model_single_abnormal.predict_on_batch(batch_b)[0]
            time_1 = time.time()
            # Temporally delete double_abnormal
            # do things of double_abnormal detect model
            # preds_double_abnormal = model_double_abnormal.predict_on_batch([batch_c, batch_b2])[0]
            time_2 = time.time()
            # do things of single count model twice
            preds_single_count_a = model_single_count.predict_on_batch([batch_a, batch_lc])
            preds_single_count_b = model_single_count.predict_on_batch([batch_b, batch_lc])
            time_3 = time.time()
            # loop over the request IDs and their corresponding set of
            # results from our model
            for i in range(batch_size):
                # Temporally delete double_abnormal
                # out = {"single_abnormal": preds_single_abnormal[i],
                #        "single_count_a": preds_single_count_a[i],
                #        "single_count_b": preds_single_count_b[i],
                #        "double_abnormal": preds_double_abnormal[i]}
                out = {"single_abnormal": preds_single_abnormal[i],
                       "single_count_a": preds_single_count_a[i],
                       "single_count_b": preds_single_count_b[i]}
                db.set(requestIDs[i], pickle.dumps(out))
            # remove the set of images from our queue
            db.ltrim(settings.QUEUE_GPUS_R1[gpu_index], len(requestIDs), -1)
            # time
            run_time = time.time() - start_time
            print("* take_goods, Batch size: {0}, use time: {1:.3f}, single_abnormal time : {2:.3f}, "
                  "double_abnormal time : {3:.3f}, 2*single_count time : {4:.3f},"
                  .format(batch_size, run_time, time_1-time_0, time_2-time_1, time_3-time_2))

        # Queue of renew_goods_abnormal
        queue_r2 = db.lrange(settings.QUEUE_GPUS_R2[gpu_index], 0, settings.BATCH_SIZE - 1)
        batch_size = len(queue_r2)
        # check to see if the batch list is None
        if batch_size > 0:
            # time
            start_time = time.time()
            requestIDs = []
            batch_b = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            # loop over the queue
            for i, q in enumerate(queue_r2):
                q = pickle.loads(q)
                # deserialize the object and obtain the input image
                image_b = db.get(q["image_b_key"])
                if image_b is not None:
                    batch_b[i] = pickle.loads(image_b)
                # update the list of image IDs
                requestIDs.append(q["request_id_diy"])
            # do things of single_abnormal detect model
            preds_single_abnormal = model_single_abnormal.predict_on_batch(batch_b)[0]
            # loop over the request IDs and their corresponding set of
            # results from our model
            for i in range(batch_size):
                out = {"single_abnormal": preds_single_abnormal[i]}
                # single count result
                db.set(requestIDs[i], pickle.dumps(out))
            # remove the set of images from our queue
            db.ltrim(settings.QUEUE_GPUS_R2[gpu_index], len(requestIDs), -1)
            # time
            run_time = time.time() - start_time
            print("* renew_goods_abnormal, Batch size: {0}, use time: {1:.3f}".format(batch_size, run_time))

        # Queue of renew_goods_count
        queue_r3 = db.lrange(settings.QUEUE_GPUS_R3[gpu_index], 0, settings.BATCH_SIZE - 1)
        batch_size = len(queue_r3)
        # check to see if the batch list is None
        if batch_size > 0:
            # time
            start_time = time.time()
            requestIDs = []
            batch_b = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            batch_lc = np.zeros((batch_size, 2 * settings.COLUMN_NUM), dtype=data_type)
            # loop over the queue
            for i, q in enumerate(queue_r3):
                q = pickle.loads(q)
                # deserialize the object and obtain the input image
                image_b = db.get(q["image_b_key"])
                if image_b is not None:
                    batch_b[i] = pickle.loads(image_b)
                lc = np.array(json.loads(q["column_lc"]))
                batch_lc[i] = lc
                # update the list of image IDs
                requestIDs.append(q["request_id_diy"])
            # do things of single count model
            preds_single_count = model_single_count.predict_on_batch([batch_b, batch_lc])
            # loop over the request IDs and their corresponding set of
            # results from our model
            for i in range(batch_size):
                out = {"single_count": preds_single_count[i]}
                # single count result
                db.set(requestIDs[i], pickle.dumps(out))
            # remove the set of images from our queue
            db.ltrim(settings.QUEUE_GPUS_R3[gpu_index], len(requestIDs), -1)
            # time
            run_time = time.time() - start_time
            print("* renew_goods_count, Batch size: {0}, use time: {1:.3f}".format(batch_size, run_time))
        # time
        loop_time = time.time() - loop_start_time
        # sleep for rest time
        if loop_time < settings.SERVER_SLEEP:
            time.sleep(settings.SERVER_SLEEP - loop_time)


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    predict_process()
