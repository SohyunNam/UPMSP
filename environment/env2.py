from environment.simulation import *

# np.random.seed(10)


class UPMSP:
    def __init__(self, num_jt=10, num_j=1000, num_m=8, log_dir=None, K=1):
        self.num_jt = num_jt
        self.num_machine = num_m
        self.jobtypes = [i for i in range(num_jt)]  # 1~10
        self.p_ij, self.p_j, self.weight = self._generating_data()
        self.num_job = num_j

        self.log_dir = log_dir

        self.jobtype_assigned = list()  # 어느 job이 어느 jobtype에 할당되는 지
        self.job_list = list()  # 모델링된 Job class를 저장할 리스트
        self.jt_dict = dict()

        self.K = K
        self.done = False
        self.tardiness = 0.0
        self.e = 0
        self.time = 0

        self.mapping = {0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"}

        self.sim_env, self.process_dict, self.source_dict, self.sink, self.routing, self.monitor = self._modeling()

    def step(self, action):
        done = False
        self.previous_time_step = self.sim_env.now
        routing_rule = self.mapping[action]

        self.routing.decision.succeed(routing_rule)
        self.routing.indicator = False

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.sink.finished_job == self.num_job:
                done = True
                self.sim_env.run()
                if self.e % 50 == 0:
                    self.monitor.save_tracer()
                # self.monitor.save_tracer()
                break

            if len(self.sim_env._queue) == 0:
                self.monitor.save_tracer()
            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        return next_state, reward, done

    def reset(self):
        self.e = self.e + 1 if self.e > 1 else 1  # episode
        self.p_ij, self.p_j, self.weight = self._generating_data()
        self.jt_dict = dict()
        self.sim_env, self.process_dict, self.source_dict, self.sink, self.routing, self.monitor = self._modeling()
        self.done = False
        self.monitor.reset()

        self.tardiness = 0

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break

            self.sim_env.step()

        return self._get_state()

    def _modeling(self):
        env = simpy.Environment()

        monitor = Monitor(self.log_dir + '/log_%d.csv'% self.e)
        # monitor = Monitor("C:/Users/sohyon/PycharmProjects/UPJSP_SH/environment/result/log_{0}.csv".format(self.e))
        process_dict = dict()
        source_dict = dict()
        time_dict = dict()  # {"JobType 0" : [pij,...], ... }
        routing = Routing(env, process_dict, source_dict, monitor, self.weight)

        # 0에서 9까지 랜덤으로 배정
        jt_dict = dict()
        self.jobtype_assigned = np.random.randint(low=0, high=10, size=self.num_job)
        for i in range(self.num_job):
            jt = self.jobtype_assigned[i]
            if "JobType {0}".format(jt) not in jt_dict.keys():
                jt_dict["JobType {0}".format(jt)] = list()
                time_dict["JobType {0}".format(jt)] = self.p_ij[jt]
            jt_dict["JobType {0}".format(jt)].append(
                Job("Job_{0}_{1}".format(jt, i), self.p_ij[jt], job_type=jt))

        self.jt_dict = copy.deepcopy(jt_dict)
        sink = Sink(env, monitor, self.jt_dict, self.num_job, source_dict, self.weight)

        for jt_name in self.jt_dict.keys():
            source_dict["Source {0}".format(int(jt_name[-1]))] = Source("Source {0}".format(int(jt_name[-1])), env,
                                                                        routing, monitor, self.jt_dict, self.p_j, self.K,
                                                                        self.num_machine)

        for i in range(self.num_machine):
            process_dict["Machine {0}".format(i)] = Process(env, "Machine {0}".format(i), sink, routing, monitor)

        return env, process_dict, source_dict, sink, routing, monitor

    def _get_state(self):
        # define 3 features
        f_1 = np.zeros(self.num_jt)
        f_2 = np.zeros(self.num_machine)
        f_3 = np.zeros(4)

        # f_1 (JOB) : the number of non-processes job in JT_j (nj)
        for jt_name in self.source_dict.keys():
            jt_idx = int(jt_name[-1])
            w_j = [1 for job in self.routing.queue.items if job.job_type == jt_idx]
            f_1[jt_idx] = sum(w_j) / len(self.routing.queue.items) if len(self.routing.queue.items) > 0 else 0.0

        # f_2 (MACHINE) : how much the time is remaining
        for i in range(self.num_machine):
            process = self.process_dict["Machine {0}".format(i)]
            f_2[i] = (process.planned_finish_time - self.sim_env.now) / self.p_ij[process.job.job_type][i] if not process.idle else 0

        # f_3 (JOB) : Tardiness Level of Jobs in Routing Queue
        job_list = copy.deepcopy(self.routing.queue.items)
        g_1 = 0
        g_2 = 0
        g_3 = 0
        g_4 = 0

        if len(job_list) > 0:
            for job in job_list:
                tightness = job.due_date - self.sim_env.now
                if tightness > np.max(self.p_ij[job.job_type]):
                    g_1 += 1
                elif (tightness > np.min(self.p_ij[job.job_type])) and (tightness <= np.max(self.p_ij[job.job_type])):
                    g_2 += 1
                elif (tightness > 0) and (tightness <= np.min(self.p_ij[job.job_type])):
                    g_3 += 1
                else:
                    g_4 += 1

            f_3[0] = g_1 / len(job_list)
            f_3[1] = g_2 / len(job_list)
            f_3[2] = g_3 / len(job_list)
            f_3[3] = g_4 / len(job_list)

        state = np.concatenate((f_1, f_2, f_3), axis=None)
        return state

    def _calculate_reward(self):
        reward = 0
        finished_jobs = copy.deepcopy(self.sink.job_list)
        for job in finished_jobs:
            jt = job.job_type
            w_j = self.weight[jt]

            tardiness = min(job.due_date - job.completion_time, 0)

            reward += w_j * tardiness

        self.sink.job_list = list()

        return reward

    def _generating_data(self):
        processing_time = [[np.random.uniform(low=1, high=20) for _ in range(self.num_machine)] for _ in range(self.num_jt)]
        p_j = [np.mean(jt_pt) for jt_pt in processing_time]
        weight = list(np.random.uniform(low=0, high=5, size=self.num_jt))

        return processing_time, p_j, weight
