import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import argparse


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)  # Set the agent in the environment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning  # Whether the agent is expected to learn
        self.Q = dict()  # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon  # Random exploration factor
        self.alpha = alpha  # Learning factor

        # TODO: Set any additional class parameters as needed

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        # TODO: Update epsilon using a decay function of your choice
        # TODO: Update additional class parameters as needed
        # TODO: If 'testing' is True, set epsilon and alpha to 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the
            environment. The next waypoint, the intersection inputs, and the deadline
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint()  # The next waypoint
        inputs = self.env.sense(self)  # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ###########
        ###########

        # NOTE: you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed
        #   constraints in order for you to learn how to adjust epsilon and alpha,
        #   and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.

        # TODO: Set 'state' as a tuple of relevant data for the agent
        state = None

        return state

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        # TODO: Calculate the maximum Q-value of all actions for a given state

        maxQ = None

        return maxQ

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        # TODO: When learning, check if the 'state' is not in the Q-table
        # TODO: If it is not, create a new dictionary for that state
        #   TODO: Then, for each action available, set the initial Q-value to 0.0

        return

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()

        # not learning means totally random action
        if not self.learning:
            action = random.choice(self.valid_actions)
        else:
            action = None

        # TODO: When learning, choose a random action with 'epsilon' probability
        # TODO: Otherwise, choose an action with the highest Q-value for the current state
        # TODO: Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        return action

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards
            when conducting learning. """

        # TODO: When learning, implement the value iteration update rule
        #   TODO: Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        return

    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()  # Get current state
        self.createQ(state)  # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action)  # Receive a reward
        self.learn(state, action, reward)  # Q-learn

        return


def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    env_flags, agent_flags, follow_flag, sim_flags, run_flags = command_line_parse()

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(**env_flags)

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, **agent_flags)

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, **follow_flag)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, **sim_flags)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(**run_flags)


def command_line_parse():
    """command line parser using argparse, obviates the need to edit code to run simulation"""
    parser = argparse.ArgumentParser(description='runs the smartcab simulation with various options',
                                     usage='smartcab/agent.py [-h] [-v]'
                                           '\n    env flags:   [-N <dummies> -g <cols> <rows>]'
                                           '\n    drive flags: [-l [-a <float> -e <float>] -D]'
                                           '\n    sim flags:   [-dLo -u <delay_secs>]'
                                           '\n    run flags:   [-t <tolerance> -n <tests>]',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='generates additional output from the simulation')
    # env flags
    environment = parser.add_argument_group('environment/world options')
    environment.add_argument('-N', '--num_dummies', type=int, default=100, metavar=('INT'),
                             help='number of dummy agents in the environment')
    environment.add_argument('-g', '--grid_size', nargs=2, type=tuple, default=(8, 6), metavar=('COLS', 'ROWS'),
                             help='controls the number of intersections = columns * rows')
    # agent flags
    driving_agent = parser.add_argument_group('driving agent options')
    driving_agent.add_argument('-l', '--learning', action='store_true',
                               help='forces the driving agent to use Q-learning')
    driving_agent.add_argument('-e', '--epsilon', type=float, default=1., metavar=('FLOAT'),
                               help='NO EFFECT without -l: value for the exploration factor')
    driving_agent.add_argument('-a', '--alpha', type=float, default=0.5, metavar=('FLOAT'),
                               help='NO EFFECT without -l: value for the learning rate')
    # follow flag
    driving_agent.add_argument('-D', '--deadline', action='store_true', dest='enforce_deadline',
                               help='enforce a deadline metric on the driving agent')
    # sim flags
    simulation = parser.add_argument_group('simulation options')
    simulation.add_argument('-u', '--update-delay', type=float, default=2., metavar=('SECONDS'),
                            help='time between actions of smartcab/environment')
    simulation.add_argument('-d', '--display', action='store_false', help='disable simulation GUI')
    simulation.add_argument('-L', '--log', action='store_true', dest='log_metrics',
                            help='log trial and simulation results to /logs')
    simulation.add_argument('-o', '--optimized', action='store_true',
                            help='change the default log file name if optimized')
    # run flags
    running = parser.add_argument_group('run-time experiment options')
    running.add_argument('-t', '--tolerance', type=float, default=0.05, metavar=('FLOAT'),
                         help='epsilon tolerance before beginning testing after exploration')
    running.add_argument('-n', '--n_test', type=int, default=0, metavar=('INT'),
                         help='number of testing trials to perform')
    return parse_flags(vars(parser.parse_args()))


def parse_flags(flags):
    """
    gives 5 separate keyword argument dicts to pass to different functions inside run()
    :param flags: dict
    :return: tuple(dict, dict, dict, dict, dict)
    :returns (environment options, agent options, deadline option, simulation setup options, simulation running options)
    """
    return tuple({k: flags[k] for k in options} for options in (
        ['verbose', 'num_dummies', 'grid_size'],
        ['learning', 'epsilon', 'alpha'],
        ['enforce_deadline'],
        ['update_delay', 'display', 'log_metrics', 'optimized'],
        ['tolerance', 'n_test']
    ))


if __name__ == '__main__':
    run()
