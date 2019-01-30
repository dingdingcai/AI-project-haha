# USAGE
# python stress_test_take_goods_simple.py

# import the necessary packages
from threading import Thread
import requests
import time
import json


def read_json(file_name):
    f = open(file_name)
    r = f.read()
    f.close()
    if not r:
        return None
    j = json.loads(r)
    return j


# initialize the server URL along with the input data
AI_SERVER_URL = " http://61.183.254.54/m3_server/predict"
IMAGE_B = "http://61.183.254.54/m3_server/static/b.jpg"
IMAGE_C = "http://61.183.254.54/m3_server/static/c.jpg"
pointmap_c = read_json("/home/ubuntu/WorkSpace/ServerSpace/m3_server/static/c_mp.json")

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 1
SLEEP_COUNT = 0.03


def call_predict_endpoint(n):
    # request_id should be identity, below is just a sample
    request_id = n
    device_id = n // 4
    shelve_level = n % 4
    # load the input image and construct the payload for the request
    request_data = {"request_id": request_id,  # request_id should be identity
                    "device_id": device_id,  # device_id should be identity
                    "shelve_level": shelve_level,  # should be one of 1, 2, 3, 4
                    "request_type": "take_goods_simple",
                    "image_b": IMAGE_B,  # image url after closing door of this action
                    "image_c": IMAGE_C,  # image url after closing door of the last action for renew_goods
                    "pointmap_c": json.dumps(pointmap_c)}
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
        print("[INFO] thread {0} OK, use time {1:.3f}, time : {2}".format(n, run_time, r['web_time']))
        print(r["predict_result"])
        # print(r["save_result"])
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
