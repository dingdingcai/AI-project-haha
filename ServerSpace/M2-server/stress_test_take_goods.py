# USAGE
# python stress_test_take_goods.py

# import the necessary packages
from threading import Thread
import requests
import time
import json


# initialize the server URL along with the input data
AI_SERVER_URL = "http://localhost/ai_server/predict"
IMAGE_A = "http://localhost/ai_server/static/a.jpg"
IMAGE_B = "http://localhost/ai_server/static/b.jpg"
# AI_SERVER_URL = "http://61.183.254.55/ai_server/predict"
# IMAGE_A = "http://img.hahabianli.com/buy/201808/13/5b71238a54318.jpg"
# IMAGE_B = "http://img.hahabianli.com/buy/201808/13/5b71239e5102e.jpg"
IMAGE_C = "http://localhost/ai_server/static/c.jpg"
# column_attribute = [0, 0, 1, 3, 0, 0, 0]  # column attribute of the last action for renew_goods
column_attribute = [0, 0, 0, 0, 0, 0, 0]  # column attribute of the last action for renew_goods

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 1
SLEEP_COUNT = 0.025


def call_predict_endpoint(n):
    # request_id should be identity, below is just a sample
    request_id = n
    device_id = n // 4
    shelve_level = n % 4
    # load the input image and construct the payload for the request
    request_data = {"request_id": request_id,  # request_id should be identity
                    "device_id": device_id,  # device_id should be identity
                    "shelve_level": shelve_level,  # should be one of 1, 2, 3, 4
                    "request_type": "take_goods",  # should be one of take_goods, renew_goods_abnormal, renew_goods_count
                    "image_a": IMAGE_A,  # image url before openning door of this action
                    "image_b": IMAGE_B,  # image url after closing door of this action
                    "image_c": IMAGE_C,  # image url after closing door of the last action for renew_goods
                    "column_attribute": json.dumps(column_attribute)}  # column attribute of the last action for renew_goods

    # time
    start_time = time.time()
    # submit the request
    try:
        response = requests.post(AI_SERVER_URL, json=request_data, timeout=15)
    except requests.exceptions.ReadTimeout:
        print('[ERROE] thread {} timeout error'.format(n))
        return None
    except requests.exceptions.ConnectionError:
        print('[ERROE] thread {} connection error'.format(n))
        return None
    if not response.status_code == requests.codes.ok:
        print("[ERROE] thread {} reponse fail code: {}".format(n, response.status_code))
        return None
    # time
    run_time = time.time() - start_time
    # read result json
    r = response.json()
    if r is None:
        print('[ERROE] thread {} server model error'.format(n))
        return None
    # ensure the request was sucessful
    if r["request_success"]:
        if run_time > 1.5:
            print("[INFO] thread {0} failed, use time {1:.3f}, time : {2}".format(n, run_time, r['web_time']))
        else:
            print("[INFO] thread {0} ok".format(n))
        print(r["predict_result"])
        print(r["save_result"])
        # print(r["web_time"])
    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED, error code {}, error des: {}".format(n, r['error_code'], r['error_description']))


# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)
