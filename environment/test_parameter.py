import os

from test_simulation import *
from calculate_tardiness import *

num_jt = 10
num_job = 1000
num_machine = 8
K = 1  # slack factor

np.random.seed(10)

def generating_data():
    processing_time = [[np.random.uniform(low=1, high=20) for _ in range(num_machine)] for _ in range(num_jt)]
    p_j = [np.mean(jt_pt) for jt_pt in processing_time]
    weight = list(np.random.uniform(low=0, high=5, size=num_jt))

    return processing_time, p_j, weight


if __name__ == "__main__":
    tard_list = list()

    jobtypes = [i for i in range(10)]

    rule = "WCOVERT"

    event_tracer_path = "./test/{0}".format(rule)
    if not os.path.exists(event_tracer_path):
        os.makedirs(event_tracer_path)
    k_list = [(i+1) * 0.5 for i in range(100)]

    opt_k = k_list[0]
    opt_mwt = 1e10
    for k in k_list:
        print("K_t = ", k)
        for idx in range(100):
            env = simpy.Environment()
            pij_data, p_j_data, w_data = generating_data()
            monitor = Monitor(event_tracer_path + '/log {0}.csv'.format(idx))
            monitor.reset()
            # monitor = Monitor("C:/Users/sohyon/PycharmProjects/UPJSP_SH/environment/result/log_{0}.csv".format(self.e))
            process_dict = dict()
            source_dict = dict()
            jt_dict = dict()  # {"JobType 0" : [Job class(), ... ], ... }
            time_dict = dict()  # {"JobType 0" : [pij,...], ... }
            routing = Routing(env, process_dict, source_dict, monitor, w_data, routing_rule=rule, k_t=k)
            routing.reset()

            # 0에서 9까지 랜덤으로 배정
            jobtype_assigned = np.random.randint(low=0, high=10, size=num_job)
            for i in range(num_job):
                jt = jobtype_assigned[i]
                if "JobType {0}".format(jt) not in jt_dict.keys():
                    jt_dict["JobType {0}".format(jt)] = list()
                    time_dict["JobType {0}".format(jt)] = pij_data[jt]
                jt_dict["JobType {0}".format(jt)].append(
                    Job("Job {0}-{1}".format(jt, i), pij_data[jt], job_type=jt))

            sink = Sink(env, monitor, jt_dict, num_job, source_dict, w_data)
            sink.reset()

            for jt_name in jt_dict.keys():
                source_dict["Source {0}".format(int(jt_name[-1]))] = Source("Source {0}".format(int(jt_name[-1])), env,
                                                                            routing, monitor, jt_dict, p_j_data, K,
                                                                            num_machine)

            for i in range(num_machine):
                process_dict["Machine {0}".format(i)] = Process(env, "Machine {0}".format(i), sink, routing, monitor)
                process_dict["Machine {0}".format(i)].reset()

            env.run()
            # monitor.save_tracer()

            mean_wt = monitor.tardiness / num_job
            tard_list.append(mean_wt)
            # print("Episode {0} | MWT = {1}".format(idx, mean_wt))

        print("k = {0}, Total Mean Weighted Tardiness = {1}".format(k, np.mean(tard_list)))

        if np.mean(tard_list) < opt_mwt:
            opt_k = k
            opt_mwt = np.mean(tard_list)

    print("Optimal k for WCOVERT = {0}, Mean Weighted Tardiness = {1}".format(opt_k, opt_mwt))