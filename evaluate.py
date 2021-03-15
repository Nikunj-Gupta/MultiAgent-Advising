from tensorboardX import SummaryWriter
from pathlib import Path
from itertools import count 
from pettingzoo.mpe import simple_spread_v2 
from dqn import DQN 
import argparse, os 


def run(args): 
    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles) 
    env.reset() 
 
    expname = args.expname + "--evaluate" 
    log_dir = Path(os.path.join(args.logdir, expname)) 
    for i in count(0):
        temp = log_dir/('run{}'.format(i)) 
        if temp.exists(): 
            pass
        else:
            writer = SummaryWriter(logdir=temp)
            log_dir = temp
            break
    obs_dim = env.observation_spaces[env.agents[0]].shape[0] 
    action_dim = env.action_spaces[env.agents[0]].n 
    agent = DQN(obs_dim, action_dim) 
    agent.load_model(args.load) 

    for episode in range(args.maxepisodes+1):
        state = env.reset() 
        state = state["agent_0"]
        episode_rewards = 0 
        for cycle in range(args.maxcycles): 
            # env.render() 
            action = agent.select_action(state) 
            action_env = {"agent_0": action[0, 0].item()}
            next_state, reward, done, _ = env.step(action_env) 
            next_state = next_state["agent_0"] 
            reward = reward["agent_0"] 
            done = done["agent_0"] 
            episode_rewards += reward 

            state = next_state
            
            if done: 
                writer.add_scalar('Episodic Reward', episode_rewards, episode)
                print("[Episode {:>5}]  Reward: {:>5}".format(episode, episode_rewards))
                episode_rewards = 0                 
                break 

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/hyperparams.yaml', help="config file name") 
    parser.add_argument("--load", type=str, default="dqn-exp/savedir/teacher/checkpoint2000--model.pth") 
    parser.add_argument("--expname", type=str, default="student-test")
    parser.add_argument("--envname", type=str, default='cn')
    parser.add_argument("--nagents", type=int, default=1) 

    parser.add_argument("--maxepisodes", type=int, default=2000) 
    parser.add_argument("--maxcycles", type=int, default=25) 

    parser.add_argument("--randomseed", type=int, default=100) 

    parser.add_argument("--logdir", type=str, default="dqn-exp/logs/", help="log directory path")
    

    args = parser.parse_args() 
    run(args=args)


