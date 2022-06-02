import os
import pandas as pd

from dqn import *
from environment.env import UPMSP
from calculate_tardiness import *


if __name__ == "__main__":
    state_size = 104
    action_size = 4

    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../test/result'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    tard_list = list()
    for i in range(100):
        env = UPMSP(log_dir=event_path + "/event_tracer_{0}.csv".format(i))
        q = Qnet(state_size, action_size)
        q.load_state_dict(torch.load(log_path + '/episode99900.pt')["model_state_dict"])

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
                break
        tardiness = cal_tard(event_path + "/event_tracer_{0}.csv".format(i))
        tard_list.append(tardiness)

    print(np.mean(tard_list))