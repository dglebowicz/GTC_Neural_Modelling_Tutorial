import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon

        return None

    def init_env(self, **env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, **env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''

        # complete the code
        self.experience_buffer[s*self.num_actions+a] = [s, a, r, s1]

        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''

        # complete the code
        
        self.Q[s,a] = self.Q[s, a] + self.alpha*(r+self.gamma*np.max(self.Q[s1,:])-self.Q[s,a])
        

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # complete the code
        self.action_count += 1
        self.action_count[s, a] = 0
            
        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        
        
        if np.sum(self.Q[s]!=0):
            a = np.argmax(self.Q[s])
        else:
            a = np.random.choice(np.arange(self.num_actions))

        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        for i in range(num_planning_updates):
            
            s = np.random.choice(self.num_states)
            a = np.random.choice(self.num_actions)
            _, _, r, s1 = self.experience_buffer[s*self.num_actions+a]
            self.Q[s, a] = self.Q[s, a] + self.alpha*(r+self.epsilon*(self.action_count[s,a]**0.5)+self.gamma*np.max(self.Q[s1,:])-self.Q[s, a])

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent:

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p
        
        return None
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])
    
    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        if s==0:
            p = np.exp(self.beta1*(self.Qnet + self.p*self.rep_a))/np.sum(np.exp(self.beta1*(self.Qnet + self.p*self.rep_a)))
            
        else:
            p = np.exp(self.beta2*self.Qtd[s])/np.sum(np.exp(self.beta2*self.Qtd[s]))
        
        a = np.random.choice(2, p=p)

        return a
    
    def _update_qtd(self, s, a, r, s1, a1):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            a1 -- next action
        '''

        
        self.Qtd[s, a] = self.Qtd[s, a] + self.alpha1 * (self.Qtd[s1, a1] - self.Qtd[s, a])
        self.Qtd[s1, a1] = self.Qtd[s1, a1] + self.alpha2 * (r -self.Qtd[s1, a1])
        self.Qtd[s, a] = self.Qtd[s, a] + self.alpha1 * self.lam * (r -self.Qtd[s1, a1])
    
    def _update_qmb(self, s, a, r, s1, a1):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            a1 -- next action
        '''

        
        if (a==0 and s1==1):
            self.trans[0] +=1
        elif (a==1 and s1==2):
            self.trans[0] +=1
        else:
            self.trans[1] +=1                
        if self.trans[0] >= self.trans[1]:
            self.T1 = np.array([0.7, 0.3])
        else:
            self.T1 = np.array([0.3, 0.7])
        for i in range(2):               
            self.Qmb[i] = self.T1[i]*np.max(self.Qtd[1])+(1-self.T1[i])*np.max(self.Qtd[2])
        
        

        return None

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
            
        # complete the code
        
        self._init_history()
        self.R = [[0.5,0.5],[0.5,0.5]]
        self.Qtd = np.zeros((3, 2))
        self.Qmb = np.zeros(2)
        self.Qnet = np.zeros(2)
        self.trans = np.zeros(2)
        self.T = np.array([[0.7, 0.3], [0.3, 0.7]])
        self.rep_a = np.zeros(2)
        

        for _ in range(num_trials):

            # choose action
            a = self._policy(0)
            # get new state
            s1 = np.random.choice(np.array([1,2]), p=self.T[a, :])
            a1 = self._policy(s1)
            # receive reward
            r = np.random.choice([0,1],p=[1-self.R[s1-1][a1],self.R[s1-1][a1]])
            
            # learning
            self._update_qtd(0, a, r, s1, a1)
            self._update_qmb(0, a, r, s1, a1)
            for j in range(2):
                self.Qnet[j] = self.w*self.Qmb[j]+(1-self.w)*self.Qtd[0, j]
            
                 
            # update history
            
            self._update_history(a, s1, r)
            self.rep_a = np.zeros(2)
            self.rep_a[a]=1

            self.R += np.random.normal(0, 0.1, size = [2, 2])
            self.R = np.clip(self.R,0.25,0.75)
            
            

        return None
    
