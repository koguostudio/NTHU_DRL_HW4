import numpy as np
import random
import torch
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class Prioritized_Replay_Buffer:
    def __init__(self, size:int = 2**18) -> None:
        self.n_data = size
        # contains self.n_data - 1 's inner nodes + root 
        self.tree = np.zeros(2*self.n_data - 1)
        #self.dead_list = deque([], maxlen=5)
        self.transitions = [0] * self.n_data
        self.current_data_pointer = 0
        self.constant = .3
        self.max_priority = 1.
        self.a = .6

        self.current_size = 0

        self.current_sampled_transitions = []
        
    def update_tree(self, priority, tree_idx):
        change_in = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        if priority > self.max_priority:
            self.max_priority = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change_in
        
    def push(self, trans):

        priority = self.max_priority
        self.transitions[self.current_data_pointer] = trans

        tree_idx = self.current_data_pointer + self.n_data - 1
        self.update_tree(priority=priority, tree_idx=tree_idx)

        self.current_data_pointer = (self.current_data_pointer + 1) % self.n_data

        if self.current_size < self.n_data:
            self.current_size += 1

    def select_leaf(self):
        total_priority = self.tree[0]

        val = random.random() * total_priority # 0~total_priority
        idx = 0
        while idx < self.n_data - 1:
            left_child = idx * 2 + 1
            right_child = left_child + 1
            if val <= self.tree[left_child]:
                idx = left_child
            elif self.tree[right_child] == .0:
                idx = left_child
                val = random.random() * self.tree[idx]
            else:
                idx = right_child
                val -= self.tree[left_child]

        return idx - (self.n_data - 1)
    def retrieve(self, batch_size):
        res = []
        res_idx = []
        for i in range(batch_size):
            while True:
                idx = self.select_leaf()
                if idx not in res_idx:
                    break

            res_idx.append(idx)
            res.append(self.transitions[res_idx[i]])

        self.current_sampled_transitions = res_idx
        return res
    
    def update_sampled_transitions_Q(self, target_Q, learning_Q):
        tds = target_Q-learning_Q

        for i, td in enumerate(tds):
            #print(np.abs(td.item()))
            tree_idx = self.current_sampled_transitions[i] + self.n_data - 1
            self.update_tree(priority=(np.abs(td.item()) + self.constant) ** self.a, tree_idx=tree_idx)


class Replay_Buffer:
    def __init__(self, buffer = deque([], 80000)) -> None:
        self.buffer = buffer
    def push(self, trans):
        # inputs are arguments of Transition
        self.buffer.append(trans)
    def retrieve(self, batch_size):
        assert len(self.buffer) >= batch_size, 'no enough buffer'

        return random.sample(self.buffer, batch_size)

class N_Step_Wrapper_RB:
    def __init__(self, replay_buffer, gamma, n_step=5) -> None:
        self.replay_buffer = replay_buffer
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, trans):
        self.n_step_buffer.append(trans)

        if len(self.n_step_buffer) < self.n_step:
            return
        
        final_next_state, final_cumulative_return, final_done = self.n_step_buffer[-1][2:]
        state, action = self.n_step_buffer[0][:2]

        for trans in reversed(list(self.n_step_buffer)[:-1]):
            next_state, reward, done = trans[2:]

            final_cumulative_return = reward + self.gamma * final_cumulative_return * (1. - float(done))
            final_next_state, final_done = (next_state, done) if done else (final_next_state, final_done)

        self.replay_buffer.push(trans=Transition(state=state, next_state=final_next_state, action=action, reward=final_cumulative_return, done=final_done))

    def retrieve(self, batch_size):
        return self.replay_buffer.retrieve(batch_size)

class N_Step_Wrapper_PER:
    def __init__(self, replay_buffer, gamma, n_step=5) -> None:
        self.replay_buffer = replay_buffer
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        #self.steps = 0
        #self.steps_queue = deque(maxlen=n_step)

        print('{}-step prioritized replay buffer created.'.format(self.n_step))

    def push(self, trans):
        self.n_step_buffer.append(trans)

        #self.steps += 1
        #self.steps_queue.append(self.steps)
        #print(self.steps_queue)

        if len(self.n_step_buffer) < self.n_step:
            return
        
        final_next_state, final_cumulative_return, final_done = self.n_step_buffer[-1][2:]
        state, action = self.n_step_buffer[0][:2]

        for transision in reversed(list(self.n_step_buffer)[:-1]):
            next_state, reward, done = transision[2:]

            final_cumulative_return = reward + self.gamma * final_cumulative_return * (1. - float(done))
            final_next_state, final_done = (next_state, done) if done else (final_next_state, final_done)

        self.replay_buffer.push(trans=Transition(state=state, next_state=final_next_state, action=action, reward=final_cumulative_return, done=final_done))

    def retrieve(self, batch_size):
        return self.replay_buffer.retrieve(batch_size)
    
    def update_sampled_transitions_Q(self, target_Q, learning_Q):
        self.replay_buffer.update_sampled_transitions_Q(target_Q, learning_Q)

class Data_Processor:
    def __init__(self, batch_size, device, buffer_type='prioritized', gamma = .99, n_step=1) -> None:

        self.buffer_type = buffer_type
        self.n_step = n_step

        if buffer_type == 'prioritized':
            if self.n_step == 1:
                self.replay_buffer = Prioritized_Replay_Buffer()
            else:
                self.replay_buffer = N_Step_Wrapper_PER(replay_buffer=Prioritized_Replay_Buffer(), gamma=gamma, n_step=n_step)
        else:
            self.replay_buffer = Replay_Buffer()

        self.device = device
        self.batch_size = batch_size

    def push_transition(self, current_observation: np.ndarray, action, next_observation:np.ndarray, reward:float, done:bool):
        current_observation = current_observation[np.newaxis, :] 
        next_observation = next_observation[np.newaxis, :]
        action = action[np.newaxis, :]
        reward = np.array([[reward]])
        done = np.array([[done]])

        self.replay_buffer.push(trans=Transition(state=current_observation, next_state=next_observation, action=action, reward=reward, done=done))

    def get_batch(self):
        transitions = self.replay_buffer.retrieve(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.concatenate(batch.state), dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(np.concatenate(batch.reward), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(np.concatenate(batch.done), dtype=torch.bool, device=self.device)
        nextState_batch = torch.tensor(np.concatenate(batch.next_state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(np.concatenate(batch.action), dtype=torch.float32, device=self.device)

        return state_batch, reward_batch, done_batch, nextState_batch, action_batch
    
    def update_priority(self, q, q_target):
        self.replay_buffer.update_sampled_transitions_Q(q_target.detach(), q.detach())


