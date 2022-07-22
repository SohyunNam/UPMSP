import vessl
# import os
import pandas as pd
from agent.dqn import *
from environment.env2 import UPMSP

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('scalar/dqn')
# writer = SummaryWriter('/output/scalar/ppo')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vessl.init()

if __name__ == "__main__":
    num_episode = 40000
    episode = 1

    score_avg = 0

    state_size = 22
    action_size = 4

    # log_path = '../result/model/dqn'
    log_path = '/output/model/dqn2'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # event_path = '../environment/result'
    event_path = '/output/event/dqn2'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = False

    env = UPMSP(log_dir=event_path)
    q = Qnet(state_size, action_size).to(device)
    q_target = Qnet(state_size, action_size).to(device)
    optimizer = optim.Adam(q.parameters(), lr=1e-5, eps=1e-06)

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
    
    validation_state = np.load('/input/validation_set2.npy')
 
    for e in range(episode, episode + num_episode + 1):
        print("episode {0} starts".format(e))
        done = False
        step = 0
        state = env.reset()
        r = list()
        loss = 0
        num_update = 0

        while not done:
            epsilon = max(0.01, 0.1-0.01*(e/200))

            step += 1

            action = q.sample_action(torch.from_numpy(state).float().to(device), epsilon)

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
                print("episode %d is finished" %(e))
                if e % 100 == 0:
                    torch.save({'episode': e,
                                'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               log_path + '/episode%d.pt' % (e))
                    print('save model...')

                break
        
        print("Start Calculate Q Value")
        q.eval()
        out = q(torch.from_numpy(validation_state).float().to(device)).cpu().detach().numpy()
        out = np.max(out, axis=1)
        avg_q = np.mean(out)
        
        print("episode: %d | reward: %.4f | Q-value: %.4f" % (e, r_epi, avg_q))
        
        vessl.log(step=e, payload={'reward': sum(r)})
        vessl.log(step=e, payload={'Q-Value': avg_q})
        vessl.log(step=e, payload={'mean weighted tardiness': env.monitor.tardiness / env.num_jobs})
        
        # writer.add_scalar("Reward/Reward", sum(r), e)
        # writer.add_scalar("Performance/Q-Value", avg_q, e)
        # writer.add_scalar("Performance/Tardiness", env.monitor.tardiness / env.num_job, e)

    # writer.close()
