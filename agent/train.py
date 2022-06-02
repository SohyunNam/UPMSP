import os
import pandas as pd
from dqn import *
from environment.env import UPMSP

if __name__ == "__main__":
    num_episode = 100000

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
    q_target = Qnet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.RMSprop(q.parameters(), lr=0.00008, alpha=0.99, eps=1e-06)

    update_interval = 20
    score = 0

    step = 0
    moving_average = list()

    for e in range(num_episode):
        done = False
        step = 0
        state = env.reset()
        r = list()

        while not done:
            epsilon = max(0.01, 0.1-0.01*(e/200))

            step += 1
            action = q.sample_action(torch.from_numpy(state).float(), epsilon)

            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)
            memory.put((state, action, reward, next_state, done))

            if memory.size() > 2000:
                train(q, q_target, memory, optimizer)

            state = next_state

            if e % update_interval == 0 and e != 0:
                q_target.load_state_dict(q.state_dict())

            if done:
                print(sum(r), epsilon)
                moving_average.append(sum(r))

                df = pd.DataFrame(moving_average)
                df.to_csv("C:/Users/sohyon/PycharmProjects/UPJSP_SH/moving_average.csv")

                if e % 100 == 0:
                    torch.save({'episode': e,
                                'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               log_path + '/episode%d.pt' % (e))
                    print('save model...')

                break

