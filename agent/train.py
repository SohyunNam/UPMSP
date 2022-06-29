import os
import pandas as pd
from dqn import *
from environment.env import UPMSP

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('scalar/')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    num_episode = 100000
    episode = 1

    score_avg = 0

    state_size = 104
    action_size = 5  # Select No action 포함

    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = False

    env = UPMSP(log_dir=event_path)
    q = Qnet(state_size, action_size).to(device)
    q_target = Qnet(state_size, action_size).to(device)
    optimizer = optim.RMSprop(q.parameters(), lr=5e-8, alpha=0.99, eps=1e-06)

    if load_model:
        ckpt = torch.load(log_path + "/episode28000.pt")
        q.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        episode = ckpt["episode"]

    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    update_interval = 20
    score = 0

    step = 0
    moving_average = list()

    for e in range(episode, episode + num_episode + 1):
        env.e = e
        done = False
        step = 0
        state = env.reset()
        r = list()
        loss = 0
        num_update = 0

        while not done:
            epsilon = max(0.01, 0.1-0.01*(e/200))

            step += 1

            available_action = None
            if env.routing.for_what == "Job":
                available_action = [1, 2, 3, 4]
            elif env.routing.for_what == "Machine":
                available_action = [i for i in range(action_size)]

            action = q.sample_action(torch.from_numpy(state).float(), epsilon, available_out=available_action)

            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)

            memory.put((state, action, reward, next_state, done))

            if memory.size() > 2000:
                loss += train(q, q_target, memory, optimizer)
                num_update += 1

            state = next_state

            if e % update_interval == 0 and e != 0:
                q_target.load_state_dict(q.state_dict())

            if done:
                if e % 100 == 0:
                    torch.save({'episode': e,
                                'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               log_path + '/episode%d.pt' % (e))
                    print('save model...')

                break
        writer.add_scalar("Reward", sum(r), e)
        avg_loss = loss/num_update if num_update > 0 else 0
        writer.add_scalar("Loss", avg_loss, e)
    writer.close()
