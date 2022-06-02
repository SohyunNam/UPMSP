import simpy
import numpy as np
import pandas as pd

from SimComponents_test import *
from calculate_tardiness import *


def read_pij(jt=10, machine=8):
    pij_data = pd.read_csv("./data/data.csv")
    pij_data = list(pij_data["P_ij"])
    pij_list = [[pij_data[machine * i + j] for j in range(machine)] for i in range(jt)]
    p_j = [np.average(pij_data[j]) for j in range(jt)]

    return pij_list, p_j


def read_weight(jt=10):
    weight_data = pd.read_csv("./data/weight.csv")
    weight_data = list(weight_data["W"])

    return weight_data


if __name__ == "__main__":
    tard_list = list()
    pij_data, p_j_data = read_pij()
    w_data = read_weight()

    rule = "RANDOM"

    event_tracer_path = "./test/{0}".format(rule)
    if not os.path.exists(event_tracer_path):
        os.makedirs(event_tracer_path)

    for i in range(10):
        num_jt = 10
        num_job = 100
        num_machine = 8

        jobtypes = [i for i in range(10)]

        env = simpy.Environment()
        event_path = event_tracer_path + "/sample_result_{0}_{1}.csv".format(rule, i)
        monitor = Monitor(event_path)

        process_dict = dict()
        source_dict = dict()
        jt_dict = dict()  # {"JobType 0" : [Job class(), ... ], ... }
        time_dict = dict()  # {"JobType 0" : [pij,...], ... }
        routing = Routing(env, process_dict, source_dict, monitor, w_data, routing_rule=rule)

        # 0에서 9까지 랜덤으로 배정
        jobtype_assigned = np.random.randint(low=0, high=10, size=num_job)
        for i in range(num_job):
            jt = jobtype_assigned[i]
            if "JobType {0}".format(jt) not in jt_dict.keys():
                jt_dict["JobType {0}".format(jt)] = list()
                time_dict["JobType {0}".format(jt)] = pij_data[jt]
            jt_dict["JobType {0}".format(jt)].append(
                Job("Job {0}-{1}".format(jt, i), pij_data[jt], job_type=jt))

        sink = Sink(env, monitor, jt_dict, num_job)

        for i in range(num_jt):
            source_dict["Source {0}".format(i)] = Source("Source {0}".format(i), env, routing, monitor, jt_dict,
                                                         p_j_data, 1)

        for i in range(num_machine):
            process_dict["Machine {0}".format(i)] = Process(env, "Machine {0}".format(i), sink, routing, monitor)

        env.run()
        monitor.save_tracer()

        mean_wt = cal_tard(event_path)
        tard_list.append(mean_wt)

    print(np.mean(tard_list))
