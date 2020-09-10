from xvfbwrapper import Xvfb
import minerl
import gym

gym.logger.set_level(40) #to get rid of "UserWarning: WARN: Box bound precision lowered by casting to float32"


def main():

    print('Hi!')
    # minerl.data.download('C:\MineRL', experiment='MineRLObtainDiamond-v0')
    # minerl.data.download('C:\MineRL', experiment='MineRLObtainDiamondVectorObf-v0')
    
    #Running an env
    # env = gym.make('MineRLObtainDiamond-v0')
    env = gym.make('MineRLTreechopVectorObf-v0')
    # env = gym.make('MineRLTreechop-v0')
    obs = env.reset()

    done = False
    while not done:
        env.render()
        action = env.action_space.sample() 
    
        # One can also take a no_op action with
        # action =env.action_space.noop()
        
        obs, reward, done, info = env.step(
            action)
    env.close()

if __name__ == "__main__":
    # vdisplay = Xvfb(width=1280, height=740, colordepth=16)
    # vdisplay.start()
    main()
    # vdisplay.stop()