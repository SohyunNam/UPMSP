import vessl

from ppo import *
from environment.env import UPMSP

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/output/scalar/ppo')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vessl.init()

if __name__ == "__main__":
    num_episode = 100000
    episode = 1

    score_avg = 0

    state_size = 104
    action_size = 4

    log_path = '/output/model/ppo'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '/output/event/ppo'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = False
    env = UPMSP(log_dir=event_path)

    model = PPO(state_size, action_size).to(device)
    num_episode = 100000

    if load_model:
        ckpt = torch.load(log_path + "/episode3300.pt")
        model.load_state_dict(ckpt["model_state_dict"])
        model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        episode = ckpt["epoch"]

    for e in range(episode, episode + num_episode + 1):
        env.e = e
        state = env.reset()
        r_epi = 0.0
        done = False

        while not done:
            for t in range(T_horizon):
                logit = model.pi(torch.from_numpy(state).float().to(device))
                prob = torch.softmax(logit, dim=-1)

                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done = env.step(action)

                model.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                r_epi += reward
                if done:
                    # print("episode: %d | reward: %.4f" % (e, r_epi))

                    if e % 50 == 0:
                        torch.save({"epoch": e,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": model.optimizer.state_dict()},
                                   log_path + "/episode%d.pt" % e)

                        # env.save_event_log(simulation_dir + "episode%d.csv" % e)

                    break

            model.train_net()
        vessl.log(step=e, payload={'reward': r_epi})
        vessl.log(step=e, payload={'mean weighted tardiness' : env.monitor.tardiness / env.num_job})
        print("episode: %d | reward: %.4f | tardiness: %.4f" % (e, r_epi, env.monitor.tardiness / env.num_job))

        writer.add_scalar("Reward/Reward", r_epi, e)
        # writer.add_scalar("Performance/Q-Value", avg_q, e)
        writer.add_scalar("Performance/Tardiness", env.monitor.tardiness / env.num_job, e)

    writer.close()