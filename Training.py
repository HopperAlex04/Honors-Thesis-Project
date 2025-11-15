import Players
import time
from GameEnvironment import CuttleEnvironment

def selfPlayTraining(p1: Players.Agent, p2: Players.Agent, episodes):
    env = CuttleEnvironment()
    actions = env.actions
    currPlayer = p1
    steps = 0

    for ep in range(episodes):
        env.reset()
        p1_states = []
        p1_actions = []
        p1_next_state = None
        p1_reward = 0
        turns = 0
        p2_actions = []
        p2_states = []
        p2_next_state = None
        p2_reward = 0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            steps += 1
            currPlayer = p1

            ob = env.get_obs()

            p1_states.append(ob)

            valid_actions = env.generateActionMask()
            p1_act = p1.getAction(ob, valid_actions, actions, steps)
            print(valid_actions)
            p1_actions.append(p1_act)
            env.updateStack(p1_act)
            depth = 1
            while env.checkResponses():
                if currPlayer == p1:
                    currPlayer = p2
                else:
                    currPlayer = p1
                env.passControl()
                #Maybe add a mode flag?
                valid_actions = env.generateActionMask(countering=True)
                ob = env.get_obs()
                response = currPlayer.getAction(ob, valid_actions, actions, steps)
                env.step(response)
                if currPlayer == p1:
                    p1_states.append(ob)
                    p1_actions.append(response)
                else:
                    p2_states.append(ob)
                    p2_actions.append(response)
                env.updateStack(response, depth)
                depth += 1
            depth = 0

            env.resolveStack()

            if currPlayer == p2:
                currPlayer = p1
                env.passControl()
            if env.stackTop() != 0:
                env.emptyStack()
                print(p1_act)
                ob, p1score, terminated, truncated = env.step(p1_act)

            if env.stackTop() == 7:
                valid_actions = env.generateActionMask()
                ob = env.get_obs()
                p1_states.append(ob)
                p1_act = p1.getAction (ob, valid_actions, actions, steps)
                p1_actions.append(p1_act)
                ob, p1score, terminated, truncated = env.step(p1_act)

            if env.stackTop()== 4:
                env.passControl()
                ob = env.get_obs()
                p2_states.append(ob)
                valid_actions = env.generateActionMask()
                p2_act = p2.getAction(ob, valid_actions, actions, steps)
                p2_actions.append(p2_act)
                ob, p2score, terminated, truncated = env.step(p2_act)
                env.passControl()

            if terminated:
                for x in range(len(p1_actions)):
                    p1.memory.push(p1_states[x], p1_actions[x], None, 1)
                for x in range(len(p2_actions)):
                    p2.memory.push(p2_states[x], p2_actions[x], None, -1)
                break
            if truncated:
                for x in range(len(p1_actions)):
                    p1.memory.push(p1_states[x], p1_actions[x], None, 0)
                for x in range(len(p2_actions)):
                    p2.memory.push(p2_states[x], p2_actions[x], None, 0)
                    break
            #env.render()

            # Push P2 stuff and then reset them

            env.emptyStack()
            env.passControl()
            env.end_turn()
            p2_next_state = env.get_obs()
            for x in range(len(p2_actions)):
                p2.memory.push(p2_states[x], p2_actions[x], p2_next_state, 0)

            p2_states = []
            p2_actions = []

            currPlayer = p2

            ob = env.get_obs()

            p2_states.append(ob)

            valid_actions = env.generateActionMask()
            p2_act = p2.getAction(ob, valid_actions, actions, steps)
            p2_actions.append(p2_act)
            env.updateStack(p2_act)
            depth = 1
            while env.checkResponses():
                if currPlayer == p2:
                    currPlayer = p1
                else:
                    currPlayer = p2
                env.passControl()
                #Maybe add a mode flag?
                valid_actions = env.generateActionMask(countering=True)
                ob = env.get_obs()
                response = currPlayer.getAction(ob, valid_actions, actions, steps)
                env.step(response)
                if currPlayer == p1:
                    p1_states.append(ob)
                    p1_actions.append(response)
                else:
                    p2_states.append(ob)
                    p2_actions.append(response)
                env.updateStack(response, depth)
                depth += 1
            depth = 0

            env.resolveStack()

            if currPlayer == p1:
                currPlayer = p2
                env.passControl()
            if env.stackTop() != 0:
                env.emptyStack()
                ob, p2score, terminated, truncated = env.step(p2_act)


            if env.stackTop() == 7:
                valid_actions = env.generateActionMask()
                ob = env.get_obs()
                p2_states.append(ob)
                p2_act = p2.getAction (ob, valid_actions, actions, steps)
                p2_actions.append(p2_act)
                ob, p2score, terminated, truncated = env.step(p2_act)

            if env.stackTop() == 4:
                env.passControl()
                ob = env.get_obs()
                p1_states.append(ob)
                valid_actions = env.generateActionMask()
                p1_act = p1.getAction(ob, valid_actions, actions, steps)
                p1_actions.append(p1_act)
                ob, p1score, terminated, truncated = env.step(p1_act)
                env.passControl()


            if terminated:
                for x in range(len(p1_actions)):
                    p1.memory.push(p1_states[x], p1_actions[x], None, 1)
                for x in range(len(p2_actions)):
                    p2.memory.push(p2_states[x], p2_actions[x], None, -1)
                break
            if truncated:
                for x in range(len(p1_actions)):
                    p1.memory.push(p1_states[x], p1_actions[x], None, 0)
                for x in range(len(p2_actions)):
                    p2.memory.push(p2_states[x], p2_actions[x], None, 0)
                    break

            env.emptyStack()
            env.end_turn()
            env.passControl()
            p1_next_state = env.get_obs()
            p1_states = []
            p1_actions = []
            for x in range(len(p1_actions)):
                p1.memory.push(p1_states[x], p1_actions[x], p1_next_state, 0)
        print(f"Episode {ep}")


