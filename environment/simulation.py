import simpy, os, random, copy
import pandas as pd
import numpy as np

from collections import OrderedDict


#region Job
class Job(object):
    def __init__(self, name, time, job_type=None, due_date=None):
        # 해당 job의 이름
        self.name = name
        # 해당 job의 작업시간
        self.processing_time = time
        self.job_type = job_type
        self.due_date = due_date

        self.completion_time = 0
#endregion


#region Source
class Source:
    def __init__(self, name, env, routing, monitor, jt_dict, p_j, K):
        self.env = env
        self.name = name
        self.routing = routing
        self.monitor = monitor
        self.p_j = p_j[int(name[-1])]  # mean of exponential distibution
        self.jobs = jt_dict["JobType {0}".format(int(name[-1]))]
        self.iat = np.random.exponential(self.p_j, size=len(self.jobs))
        self.len_jobs = len(copy.deepcopy(self.jobs))

        # set duedate
        start_time = 0
        self.due_date = list()
        for i in range(len(self.jobs)):
            start_time += np.random.exponential(1/self.p_j)
            self.jobs[i].due_date = start_time + self.p_j * K
            self.due_date.append(start_time + self.p_j * K)

        self.env.process(self.generate())

    def generate(self):
        while len(self.jobs):
            job = self.jobs.pop(0)
            yield self.env.timeout(np.random.exponential(1/self.p_j))

            self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Created", job=job.name)
            print("{0} is create, {1}/{2}".format(job.name, self.len_jobs - len(self.jobs), self.len_jobs))
            self.routing.queue.put(job)
            self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Put in Routing Class", job=job.name,
                                queue=len(self.routing.queue.items))
            self.routing.created += 1
            self.env.process(self.routing.run(location="Source"))
            print("{0} is transferred to Routing class at {1}".format(job.name, round(self.env.now, 2)))


#region Process
class Process:
    def __init__(self, env, name, sink, routing, monitor):
        self.env = env
        self.name = name
        self.sink = sink
        self.routing = routing
        self.monitor = monitor

        self.queue = simpy.Store(env)
        self.idle = True
        self.job = None
        self.planned_finish_time = 0

        env.process(self.run())

    def run(self):
        while True:
            self.job = yield self.queue.get()
            self.idle = False
            print("{0} starts working at {1}".format(self.job.name, round(self.env.now)))
            self.monitor.record(time=self.env.now, jobtype=self.job.job_type, event="Work Start", job=self.job.name,
                                machine=self.name)
            processing_time = self.job.processing_time[int(self.name[-1])]
            self.planned_finish_time = self.env.now + processing_time
            yield self.env.timeout(processing_time)
            self.job.completion_time = self.env.now
            print("{0} finishes working at {1}".format(self.job.name, round(self.env.now)))
            self.monitor.record(time=self.env.now, jobtype=self.job.job_type, event="Work Finish", job=self.job.name,
                                machine=self.name)

            self.sink.put(self.job)
            self.idle = True
            self.job = None

            if len(self.routing.queue.items) > 0:
                self.monitor.record(time=self.env.now, event="Request Routing for Job", machine=self.name,
                                    queue=len(self.routing.queue.items))
                yield self.env.process(self.routing.run(location=self.name))
            elif (len(self.queue.items) == 0) and (self.routing.created == self.sink.end_num):
                break

#endregion

#region Routing
class Routing:
    def __init__(self, env, process_dict, source_dict, monitor, weight):
        self.env = env
        self.process_dict = process_dict
        self.source_dict = source_dict
        self.monitor = monitor
        self.weight = weight

        self.created = 0

        self.queue = simpy.FilterStore(env)
        self.waiting = env.event()

        self.indicator = False
        self.decision = False

        self.idle = False
        self.job = None

        self.mapping = {0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"}

    def run(self, location="Source"):
        if location == "Source":  # job -> machine 선택
            machine_idle = [machine.idle for machine in self.process_dict.values()]
            if any(machine_idle):
                job = yield self.queue.get()
                print("Routing Logic runs at {0} for Source".format(round(self.env.now, 2)))

                self.indicator = True
                self.decision = self.env.event()
                routing_rule = yield self.decision
                self.decision = None
                self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Routing Start", job=job.name,
                                    memo="{0},  machine 선택".format(routing_rule))
                # routing_rule_number = np.random.randint(low=0, high=4)
                # routing_rule = self.mapping[routing_rule_number]

                if routing_rule == "WSPT":
                    next_machine = yield self.env.process(self.WSPT(location=location, idle=machine_idle, job=job))
                elif routing_rule == "WMDD":
                    next_machine = yield self.env.process(self.WMDD(location=location, idle=machine_idle, job=job))
                elif routing_rule == "ATC":
                    next_machine = yield self.env.process(self.ATC(location=location, idle=machine_idle, job=job))
                else:
                    next_machine = yield self.env.process(self.WCOVERT(location=location, idle=machine_idle, job=job))

                self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Routing Finish", job=job.name,
                                    machine="Machine {0}".format(next_machine))

                print("Routing Logic made {0} deliver to {1} at {2}".format(job.name, "Machine {0}".format(next_machine), round(self.env.now, 2)))
                self.process_dict["Machine {0}".format(next_machine)].queue.put(job)

        else:  # machine -> job 선택
            print("Routing Logic runs at {0} for {1}".format(round(self.env.now, 2), location))
            if len(self.queue.items) > 0:
                self.indicator = True
                # routing_rule_number = np.random.randint(low=0, high=4)
                # routing_rule = self.mapping[routing_rule_number]
                self.decision = self.env.event()
                routing_rule = yield self.decision
                self.decision = None

                self.monitor.record(time=self.env.now, event="Routing Start", machine=location, memo="{0} Job 선택".format(routing_rule))
                if routing_rule == "WSPT":
                    next_job = yield self.env.process(self.WSPT(location=location))
                elif routing_rule == "WMDD":
                    next_job = yield self.env.process(self.WMDD(location=location))
                elif routing_rule == "ATC":
                    next_job = yield self.env.process(self.ATC(location=location))
                else:
                    next_job = yield self.env.process(self.WCOVERT(location=location))

                print("Routing Logic made {0} deliver to {1} at {2}".format(next_job.name, location, round(self.env.now, 2)))

                self.monitor.record(time=self.env.now, jobtype=next_job.job_type, event="Routing Finish",
                                    job=next_job.name, machine=location)

                self.process_dict[location].queue.put(next_job)

    def WSPT(self, location="Source", idle=None, job=None):
        if location == "Source":  # job -> machine 선택 => output : machine index
            min_processing_time = 1e10
            min_machine_idx = None
            jt = job.job_type

            for idx in range(len(idle)):
                if idle[idx]:
                    wpt = job.processing_time[idx] / self.weight[jt]
                    if wpt < min_processing_time:
                        min_processing_time = wpt
                        min_machine_idx = idx

            print("WSPT, Machine = {0}".format(min_machine_idx))
            return min_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            min_processing_time = 1e10
            min_job_name = None

            for job_q in job_list:
                jt = job_q.job_type
                wpt = job_q.processing_time[int(location[-1])] / self.weight[jt]

                if wpt < min_processing_time:
                    min_processing_time = wpt
                    min_job_name = job_q.name

            next_job = yield self.queue.get(lambda x: x.name == min_job_name)
            print("WSPT, Job = {0}".format(next_job.name))
            return next_job

    def WMDD(self, location="Source", idle=None, job=None):
        if location == "Source":  # job -> machine 선택 => output : machine index
            min_wdd = 1e10
            min_machine_idx = None
            jt = job.job_type

            for idx in range(len(idle)):
                if idle[idx]:
                    wdd = max(job.processing_time[idx], job.due_date - self.env.now) / self.weight[jt]
                    if wdd < min_wdd:
                        min_wdd = wdd
                        min_machine_idx = idx
            print("WMDD, Machine = {0}".format(min_machine_idx))
            return min_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            min_wdd = 1e10
            min_job_name = None

            for job_q in job_list:
                jt = job_q.job_type
                wdd = max(job_q.processing_time[int(location[-1])], job_q.due_date - self.env.now) / self.weight[jt]

                if wdd < min_wdd:
                    min_wdd = wdd
                    min_job_name = job_q.name

            next_job = yield self.queue.get(lambda x: x.name == min_job_name)
            print("WMDD, Job = {0}".format(next_job.name))
            return next_job

    def ATC(self, location="Source", idle=None, job=None):
        h = 2.3
        if location == "Source":  # job -> machine 선택 => output : machine index
            max_wa = -1
            max_machine_idx = None
            jt = job.job_type
            non_processed_job = self.source_dict["Source {0}".format(jt)].jobs
            temp = list()
            for idx in range(len(idle)):
                if idle[idx]:
                    if len(non_processed_job) > 0:
                        p = np.average([non_job.processing_time[idx] for non_job in non_processed_job])
                        wa = self.weight[jt] * np.exp(
                            -(max(job.due_date - job.processing_time[idx] - self.env.now, 0) / (h * p))) / \
                             job.processing_time[idx]
                        if wa > max_wa:
                            max_wa = wa
                            max_machine_idx = idx
                    else:
                        temp.append(idx)

            if len(temp) > 0:
                max_machine_idx = random.choice(temp)

            print("ATC, Machine = {0}".format(max_machine_idx))
            return max_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            max_wa = -1
            max_job_name = None

            temp = list()
            for job_q in job_list:
                jt = job_q.job_type
                non_processed_job = self.source_dict["Source {0}".format(jt)].jobs

                if len(non_processed_job) > 0:
                    p = np.average([non_job.processing_time[int(location[-1])] for non_job in non_processed_job])
                    wa = self.weight[jt] * np.exp(
                        -(max(job_q.due_date - job_q.processing_time[int(location[-1])] - self.env.now, 0) / (h * p))) / \
                         job_q.processing_time[int(location[-1])]
                    if wa > max_wa:
                        max_wa = wa
                        max_job_name = job_q.name
                else:
                    temp.append(job_q.name)

            if max_job_name is None:
                max_job_name = random.choice(temp)

            next_job = yield self.queue.get(lambda x: x.name == max_job_name)
            print("ATC, Job = {0}".format(next_job.name))
            return next_job

    def WCOVERT(self, location="Source", idle=None, job=None):
        k_t = 2.3
        if location == "Source":  # job -> machine 선택 => output : machine index
            max_wt = -1
            max_machine_idx = None
            jt = job.job_type
            for idx in range(len(idle)):
                if idle[idx]:
                    p_ij = job.processing_time[idx]
                    temp_wt = max(job.due_date - p_ij - self.env.now, 0)
                    temp_wt = temp_wt / (k_t * p_ij)
                    temp_wt = max(1 - temp_wt, 0)
                    wt = self.weight[jt] * temp_wt / p_ij

                    if wt > max_wt:
                        max_wt = wt
                        max_machine_idx = idx

            print("WCOVERT, Machine = {0}".format(max_machine_idx))
            return max_machine_idx

        else:  # machine -> job 선택 => output : job
            job_list = list(copy.deepcopy(self.queue.items))
            max_wt = -1
            max_job_name = None

            for job_q in job_list:
                jt = job_q.job_type
                p_ij = job_q.processing_time[int(location[-1])]
                wt = self.weight[jt] * np.exp(1 - (max(job_q.due_date - p_ij - self.env.now, 0) / (k_t * p_ij))) / p_ij

                if wt > max_wt:
                    max_wt = wt
                    max_job_name = job_q.name

            next_job = yield self.queue.get(lambda x: x.name == max_job_name)
            print("WCOVERT, Job = {0}".format(next_job.name))
            return next_job

#endregion

#region Sink
class Sink:
    def __init__(self, env, monitor, jt_dict, end_num):
        self.env = env
        self.monitor = monitor
        self.jt_dict = jt_dict
        self.idle = False
        self.end_num = end_num

        # JobType 별 작업이 종료된 Job의 수
        self.finished = {jt: 0 for jt in self.jt_dict.keys()}
        self.finished_job = 0

        self.job_list = list()

    def put(self, job):
        print("{0} finished its work at {1}".format(job.name, round(self.env.now, 2)))
        self.finished["JobType {0}".format(job.job_type)] += 1  # jobtype 별 종료 개수
        self.finished_job += 1  # 전체 종료 개수
        self.monitor.record(time=self.env.now, jobtype=job.job_type, event="Completed", job=job.name,
                            memo=max(self.env.now - job.due_date, 0))
        self.job_list.append(job)

#endregion

#region Monitor
class Monitor:
    def __init__(self, filepath):
        self.time = list()
        self.jobtype = list()
        self.event = list()
        self.job = list()
        self.machine = list()
        self.queue = list()
        self.memo = list()

        self.filepath = filepath

    def record(self, time=None, jobtype=None, event=None, job=None, machine=None, queue=None, memo=None):
        self.time.append(round(time, 2))
        self.jobtype.append(jobtype)
        self.event.append(event)
        self.job.append(job)
        self.machine.append(machine)
        self.queue.append(queue)
        self.memo.append(memo)

    def save_tracer(self):
        event_tracer = pd.DataFrame(columns=["Time", "JobType", "Event", "Job", "Machine", "Queue", "Memo"])
        event_tracer["Time"] = self.time
        event_tracer["JobType"] = self.jobtype
        event_tracer["Event"] = self.event
        event_tracer["Job"] = self.job
        event_tracer["Machine"] = self.machine
        event_tracer["Queue"] = self.queue
        event_tracer["Memo"] = self.memo
        event_tracer.to_csv(self.filepath, encoding='utf-8-sig')
        print(self.filepath)

    def reset(self):
        self.time = list()
        self.time = list()
        self.jobtype = list()
        self.event = list()
        self.job = list()
        self.machine = list()
        self.memo = list()

#endregion