import os
import pandas as pd
from dqn import *
from environment.env import UPMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    num_episode = 11
    episode = 1

    score_avg = 0

    state_size = 104
    action_size = 4

    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    env = UPMSP(log_dir=event_path)
    q = Qnet(state_size, action_size)
    q.load_state_dict(torch.load(log_path + '/episode30300.pt')["model_state_dict"])

    tard_list = list()
    for i in range(100):
        env.e = i
        step = 0
        done = False
        state = env.reset()
        r = list()

        while not done:
            epsilon = 0
            step += 1
            action = q.sample_action(torch.from_numpy(state).float(), epsilon)

            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)
            state = next_state

            if done:
                env.monitor.save_tracer()
                break

        mean_wt = env.monitor.tardiness / env.num_job
        tard_list.append(mean_wt)
        print("Episode {0} | MWT = {1}".format(i, mean_wt))

    print("Total Mean Weighted Tardiness = ", np.mean(tard_list))